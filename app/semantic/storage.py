from semantic.graph_db import get_driver
from semantic.normalizer import normalize_triplet
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


def get_current_value(user_id: str, entity1: str, relationship_type: str):
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (e1:Entity {{name: $entity1, user_id: $user_id}})-[r:{relationship_type}]->(e2:Entity)
            WHERE r.invalid_at IS NULL
            RETURN e2.name AS value
            """,
            {"entity1": entity1, "user_id": user_id}
        )
        record = result.single()
        return record["value"] if record else None


def get_all_current_values(user_id: str, entity1: str, relationship_type: str):
    """Used for multi-valued relationships — check ALL current values, not just one."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (e1:Entity {{name: $entity1, user_id: $user_id}})-[r:{relationship_type}]->(e2:Entity)
            WHERE r.invalid_at IS NULL
            RETURN e2.name AS value
            """,
            {"entity1": entity1, "user_id": user_id}
        )
        return [record["value"] for record in result]


def get_established_valuedness(user_id: str, relationship_type: str):
    """Check if this relationship type has been used before for this user — if so, reuse its established single/multi-valued nature."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (:Entity {{user_id: $user_id}})-[r:{relationship_type}]->(:Entity)
            RETURN r.is_single_valued AS is_single_valued
            LIMIT 1
            """,
            {"user_id": user_id}
        )
        record = result.single()
        return record["is_single_valued"] if record else None


def store_triplet(user_id: str, entity1: str, relationship: str, entity2: str,
                confidence: float, importance_category: str, importance_score: float,
                is_single_valued: bool = True, source: str = None) -> dict:

    normalized = normalize_triplet(user_id, entity1, relationship, entity2)
    entity1 = normalized["entity1"]
    relationship = normalized["relationship"]
    entity2 = normalized["entity2"]

    driver = get_driver()
    relationship_type = relationship.upper()

    # ---- KEY FIX: if this relationship type was already established before, reuse its valuedness ----
    established = get_established_valuedness(user_id, relationship_type)
    if established is not None:
        is_single_valued = established
    # else: this is the first time we see this relationship type — trust the LLM's guess, and it becomes "established" once stored below

    if is_single_valued:
        existing_value = get_current_value(user_id, entity1, relationship_type)
        if existing_value and existing_value.lower() == entity2.lower():
            return {"stored": None, "invalidated_conflicts": [], "noop": True,
                    "reason": f"Fact already exists and is current: {entity1} {relationship} {entity2}"}
        conflicts = check_and_invalidate_conflict(user_id, entity1, relationship_type, entity2)
    else:
        existing_values = get_all_current_values(user_id, entity1, relationship_type)
        if any(v.lower() == entity2.lower() for v in existing_values):
            return {"stored": None, "invalidated_conflicts": [], "noop": True,
                    "reason": f"Fact already exists: {entity1} {relationship} {entity2}"}
        conflicts = []

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
                is_single_valued: $is_single_valued,
                last_accessed_at: null,
                access_count: 0
            }}]->(e2)
            RETURN e1, r, e2
            """,
            {
                "entity1": entity1, "entity2": entity2, "user_id": user_id,
                "rel_id": str(uuid.uuid4()),
                "valid_at": datetime.now(timezone.utc).isoformat(),
                "confidence": confidence,
                "importance_category": importance_category,
                "importance_score": importance_score,
                "source": source,
                "is_single_valued": is_single_valued
            }
        )
        record = result.single()

    return {"stored": record, "invalidated_conflicts": [c["old_value"] for c in conflicts], "noop": False}