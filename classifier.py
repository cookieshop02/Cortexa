import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CLASSIFIER_PROMPT = """You are a memory classification system. Given a piece of text, identify all DISTINCT units within it.

For each distinct unit, classify it as:
- "episodic": a specific event that happened (what, when)
- "semantic": a general fact, preference, or relationship (not tied to one moment)
- "procedural": a how-to, process, or skill
- "query": a QUESTION the user is asking, wanting an answer FROM their stored memories

For "query" type units ONLY, also include a "needs" field:
- "episodic": the question is about a specific past event/moment (e.g. "when did I last see a movie?")
- "semantic": the question is about a standing fact/preference (e.g. "what's my favorite color?")
- "both": the question could relate to either

Return ONLY valid JSON, no other text, in this exact format:
{
  "units": [
    {
      "content": "the extracted text for this unit",
      "type": "episodic" | "semantic" | "procedural" | "query",
      "importance_category": "low" | "medium" | "high",
      "importance_score": 0.0 to 1.0,
      "confidence": 0.0 to 1.0,
      "needs": "episodic" | "semantic" | "both"   (ONLY include this field if type is "query")
    }
  ]
}

Example 1:
Input: "User complained yesterday about late delivery. User generally prefers vegetarian food."
Output: {
  "units": [
    {"content": "User complained yesterday about late delivery", "type": "episodic", "importance_category": "medium", "importance_score": 0.6, "confidence": 0.95},
    {"content": "User generally prefers vegetarian food", "type": "semantic", "importance_category": "high", "importance_score": 0.8, "confidence": 0.9}
  ]
}

Example 2 (store + query together):
Input: "I saw the Starlight movie today. When was the last time I saw a movie?"
Output: {
  "units": [
    {"content": "User saw the Starlight movie today", "type": "episodic", "importance_category": "low", "importance_score": 0.4, "confidence": 0.9},
    {"content": "When was the last time the user saw a movie?", "type": "query", "importance_category": "low", "importance_score": 0.3, "confidence": 0.9, "needs": "episodic"}
  ]
}

Example 3 (pure query):
Input: "What is my favorite color?"
Output: {
  "units": [
    {"content": "What is the user's favorite color?", "type": "query", "importance_category": "low", "importance_score": 0.3, "confidence": 0.9, "needs": "semantic"}
  ]
}

Now classify this text:
"""


def classify_text(text: str) -> list[dict]:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": CLASSIFIER_PROMPT + text}]
    )
    raw_output = response.choices[0].message.content.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        raw_output = raw_output.replace("json", "", 1).strip()

    parsed = json.loads(raw_output)
    return parsed["units"]