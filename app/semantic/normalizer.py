import os
import json
from groq import Groq
from dotenv import load_dotenv
from semantic.graph_db import get_driver

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_known_relationships(user_id: str) -> list[str]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (:Entity {user_id: $user_id})-[r]->(:Entity)
            RETURN DISTINCT type(r) AS rel_type
            """,
            {"user_id": user_id}
        )
        return [record["rel_type"] for record in result]


def get_known_entities(user_id: str) -> list[str]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity {user_id: $user_id})
            RETURN DISTINCT e.name AS entity_name
            """,
            {"user_id": user_id}
        )
        return [record["entity_name"] for record in result]


def normalize_value(new_value: str, known_values: list[str], value_type: str) -> str:
    if not known_values:
        return new_value

    if new_value.lower() in [k.lower() for k in known_values]:
        # Exact match already, just return the canonical known version
        for k in known_values:
            if k.lower() == new_value.lower():
                return k

    prompt = f"""Given this list of known {value_type}: {known_values}

Does "{new_value}" mean the same thing as any item in this list?
If yes, respond with ONLY the exact matching item from the list (copy it exactly).
If no, respond with exactly: NEW"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()

    if answer == "NEW" or answer not in known_values:
        return new_value
    return answer


def normalize_triplet(user_id: str, entity1: str, relationship: str, entity2: str) -> dict:
    known_entities = get_known_entities(user_id)
    known_relationships = get_known_relationships(user_id)

    normalized_entity1 = normalize_value(entity1, known_entities, "entities")
    normalized_relationship = normalize_value(relationship.lower(),[r.lower() for r in known_relationships],"relationships")
    normalized_entity2 = normalize_value(entity2, known_entities, "entities")

    return {
        "entity1": normalized_entity1,
        "relationship": normalized_relationship,
        "entity2": normalized_entity2
    }