import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

QUERY_PARSE_PROMPT = """You are a semantic query understanding system. Given a user's question, extract:

1. "entity1": the subject being asked about (usually "User")
2. "relationship": the relationship type being asked about, in snake_case (e.g. "lives_in", "prefers")
3. "wants_history": true if the user is asking about the PAST/PREVIOUS state (e.g. "used to", "before", "earlier", "purana"), false if asking about the CURRENT state
4. "topic_hint": if the question mentions a specific category/topic 
   (e.g. "color", "food", "flavor"), extract that word. Otherwise null.

Return ONLY valid JSON in this format:
{
  "entity1": "User",
  "relationship": "lives_in",
  "wants_history": false,
  "topic_hint": "color" or null
}

"""


def parse_semantic_query(query: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": QUERY_PARSE_PROMPT + query}]
    )
    raw_output = response.choices[0].message.content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()
    return json.loads(raw_output)