from graph_db import get_driver
from datetime import datetime, timezone
import uuid


def check_and_invalidate_conflict(user_id: str, entity1: str, relationship_type: str, new_entity2: str):
    driver = get_driver()

    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (e1:Entity {{name: $entity1, user_id: $user_id}})-[r:{relationship_type}]->(e2:Entity)
            WHERE r.invalid_at IS NULL AND e2.name <> $new_entity2
            RETURN e2.name AS old_value, r.id AS rel_id
            """,
            {"entity1": entity1, "user_id": user_id, "new_entity2": new_entity2}
        )
        conflicts = [record for record in result]

        for conflict in conflicts:
            session.run(
                f"""
                MATCH (e1:Entity {{name: $entity1, user_id: $user_id}})-[r:{relationship_type} {{id: $rel_id}}]->(e2:Entity)
                SET r.invalid_at = $invalid_at
                """,
                {
                    "entity1": entity1,
                    "user_id": user_id,
                    "rel_id": conflict["rel_id"],
                    "invalid_at": datetime.now(timezone.utc).isoformat()
                }
            )
        return conflicts


def store_triplet(user_id: str, entity1: str, relationship: str, entity2: str,
                confidence: float, importance_category: str, importance_score: float,
                source: str = None) -> dict:

    driver = get_driver()
    relationship_type = relationship.upper()

    # Step 1: Check for conflicting current facts, and invalidate them
    conflicts = check_and_invalidate_conflict(user_id, entity1, relationship_type, entity2)

    # Step 2: Store the new fact as current
    with driver.session() as session:
        result = session.run(
            f"""
            MERGE (e1:Entity {{name: $entity1, user_id: $user_id}})
            MERGE (e2:Entity {{name: $entity2, user_id: $user_id}})
            CREATE (e1)-[r:{relationship_type} {{
                id: $rel_id,
                user_id: $user_id,
                valid_at: $valid_at,
                invalid_at: null,
                confidence: $confidence,
                importance_category: $importance_category,
                importance_score: $importance_score,
                source: $source,
                last_accessed_at: null,
                access_count: 0
            }}]->(e2)
            RETURN e1, r, e2
            """,
            {
                "entity1": entity1,
                "entity2": entity2,
                "user_id": user_id,
                "rel_id": str(uuid.uuid4()),
                "valid_at": datetime.now(timezone.utc).isoformat(),
                "confidence": confidence,
                "importance_category": importance_category,
                "importance_score": importance_score,
                "source": source
            }
        )
        record = result.single()

    return {
        "stored": record,
        "invalidated_conflicts": [c["old_value"] for c in conflicts]
    }
    #relationship type doesn't accepted as parameter that's why we have to format in in the query string directly.