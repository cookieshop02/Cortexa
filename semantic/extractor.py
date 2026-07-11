import os
import json
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACTION_PROMPT = """ You are a semantic fact extractor system. Given a piece of text (already identified as a semantic/standing fact, not a one-time event), extract it as a structured triplet.
Return ONLY valid JSON in this format:
{
  "entity1": "the subject (usually 'User' unless another entity is named)",
  "relationship": "a short snake_case relationship type, e.g. 'lives_in', 'prefers', 'works_at'",
  "entity2": "the object/value of the relationship",
  "confidence": 0.0 to 1.0,
  "importance_category": "low" | "medium" | "high",
  "importance_score": 0.0 to 1.0
}

Example:
Input: "User generally prefers vegetarian food"
Output: {
  "entity1": "User",
  "relationship": "prefers",
  "entity2": "vegetarian food",
  "confidence": 0.9,
  "importance_category": "high",
  "importance_score": 0.8
}

Now extract this text:
"""

def extract_triplet(text: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT + text}]
    )

    raw_output = response.choices[0].message.content.strip()
    
    if raw_output.startswith("'''"):
        raw_output = raw_output.strip("'")
        raw_output = raw_output.replace("json", "", 1).strip()

    return json.loads(raw_output)