import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CLASSIFIER_PROMPT = """You are a memory classification system. Given a piece of text, identify all DISTINCT memory units within it.

For each distinct unit, classify it as:
- "episodic": a specific event that happened (what, when)
- "semantic": a general fact, preference, or relationship (not tied to one moment)
- "procedural": a how-to, process, or skill

Return ONLY valid JSON, no other text, in this exact format:
{
  "units": [
    {
      "content": "the extracted text for this unit",
      "type": "episodic" | "semantic" | "procedural",
      "importance_category": "low" | "medium" | "high",
      "importance_score": 0.0 to 1.0,
      "confidence": 0.0 to 1.0
    }
  ]
}

Example:
Input: "User complained yesterday about late delivery. User generally prefers vegetarian food."
Output: {
  "units": [
    {"content": "User complained yesterday about late delivery", "type": "episodic", "importance_category": "medium", "importance_score": 0.6, "confidence": 0.95},
    {"content": "User generally prefers vegetarian food", "type": "semantic", "importance_category": "high", "importance_score": 0.8, "confidence": 0.9}
  ]
}

Now classify this text:
"""

def classify_text(text:str) -> list[dict]:
    response = client.chat.completions.create(model = "llama-3.3-70b-versatile", messages = [{"role": "user", "content": CLASSIFIER_PROMPT + text}])
    raw_output = response.choices[0].message.content.strip()

    if raw_output.startswith("'''"):
        raw_output = raw_output.strip("'")
        raw_output = raw_output.replace("json", "", 1).strip()

    parsed = json.loads(raw_output)
    return parsed["units"]