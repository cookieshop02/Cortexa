from semantic.graph_db import get_driver
from datetime import datetime, timezone


def get_current_fact(user_id: str, entity1: str, relationship_type: str):
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (e1:Entity {{name: $entity1, user_id: $user_id}})-[r:{relationship_type}]->(e2:Entity)
            WHERE r.invalid_at IS NULL
            SET r.last_accessed_at = $now, r.access_count = r.access_count + 1
            RETURN e2.name AS value, r.confidence AS confidence, r.valid_at AS valid_at
            """,
            {"entity1": entity1, "user_id": user_id, "now": datetime.now(timezone.utc).isoformat()}
        )
        return [dict(record) for record in result]


def get_historical_facts(user_id: str, entity1: str, relationship_type: str):
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (e1:Entity {{name: $entity1, user_id: $user_id}})-[r:{relationship_type}]->(e2:Entity)
            WHERE r.invalid_at IS NOT NULL
            SET r.last_accessed_at = $now, r.access_count = r.access_count + 1
            RETURN e2.name AS value, r.valid_at AS valid_at, r.invalid_at AS invalid_at
            ORDER BY r.valid_at DESC
            """,
            {"entity1": entity1, "user_id": user_id, "now": datetime.now(timezone.utc).isoformat()}
        )
        return [dict(record) for record in result]


def search_semantic_memory(user_id: str, query: str):
    from semantic.query_parser import parse_semantic_query

    parsed = parse_semantic_query(query)
    entity1 = parsed["entity1"]
    relationship_type = parsed["relationship"].upper()

    if parsed["wants_history"]:
        return get_historical_facts(user_id, entity1, relationship_type)
    else:
        return get_current_fact(user_id, entity1, relationship_type)