from semantic.graph_db import get_driver
from datetime import datetime, timezone

from embeddings import generate_embedding
import numpy as np


def get_current_fact(user_id: str, entity1: str, relationship_type: str, query_text: str = None):
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
        records = [dict(record) for record in result]

    # Agar 0 ya 1 hi match mila, ya query_text nahi diya — seedha return karo
    if len(records) <= 1 or not query_text:
        return records

    # MULTIPLE current-values mile — ab topic-disambiguation zaroori hai
    query_emb = np.array(generate_embedding(query_text))
    scored = []
    for rec in records:
        val_emb = np.array(generate_embedding(rec["value"]))
        similarity = float(np.dot(query_emb, val_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(val_emb)))
        scored.append((similarity, rec))

    scored.sort(key=lambda x: -x[0])
    best_similarity, best_record = scored[0]

    THRESHOLD = 0.35  # tune karna padega real usage se
    if best_similarity > THRESHOLD:
        return [best_record]

    return records  # agar clearly best na mile, sab return karo (ambiguous case)

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

from semantic.query_parser import parse_semantic_query

def search_semantic_memory(user_id: str, query: str):
    parsed = parse_semantic_query(query)
    entity1 = parsed["entity1"]
    relationship_type = parsed["relationship"].upper()

    if parsed["wants_history"]:
        return get_historical_facts(user_id, entity1, relationship_type)
    else:
        return get_current_fact(user_id, entity1, relationship_type, query_text=query)


def _filter_by_topic(results: list, topic_hint: str, threshold: float = 0.3):
    if not topic_hint or not results:
        return results
    
    topic_embedding = generate_embedding(topic_hint)
    filtered = []
    for r in results:
        value_embedding = generate_embedding(r["value"])
        similarity = np.dot(topic_embedding, value_embedding)  # cosine-ish similarity
        if similarity > threshold:
            filtered.append(r)
    return filtered if filtered else results  # fallback to all if nothing matches