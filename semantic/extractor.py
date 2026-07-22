import os
import json
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACTION_PROMPT = """You are a semantic fact extractor system. Given a piece of text (already identified as a semantic/standing fact, not a one-time event), extract it as a structured triplet.

Also determine if this relationship type is:
- SINGLE-VALUED: only one value can be true at a time, a new value REPLACES the old one 
  (e.g. lives_in, works_at, married_to, current_job)
- MULTI-VALUED: multiple values can be true at the same time, a new value ADDS to existing ones, does not replace 
  (e.g. prefers, likes, owns, knows, has_visited)

Return ONLY valid JSON in this format:
{
  "entity1": "the subject (usually 'User' unless another entity is named)",
  "relationship": "a short snake_case relationship type, e.g. 'lives_in', 'prefers', 'works_at'",
  "entity2": "the object/value of the relationship",
  "confidence": 0.0 to 1.0,
  "importance_category": "low" | "medium" | "high",
  "importance_score": 0.0 to 1.0,
  "is_single_valued": true or false
}

Example 1 (single-valued):
Input: "User now lives in Bangalore"
Output: {
  "entity1": "User",
  "relationship": "lives_in",
  "entity2": "Bangalore",
  "confidence": 0.9,
  "importance_category": "high",
  "importance_score": 0.8,
  "is_single_valued": true
}

Example 2 (multi-valued):
Input: "User prefers blue color"
Output: {
  "entity1": "User",
  "relationship": "prefers",
  "entity2": "blue color",
  "confidence": 0.9,
  "importance_category": "medium",
  "importance_score": 0.6,
  "is_single_valued": false
}

Now extract this text:
"""


def extract_triplet(text: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT + text}]
    )

    raw_output = response.choices[0].message.content.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        raw_output = raw_output.replace("json", "", 1).strip()

    return json.loads(raw_output)