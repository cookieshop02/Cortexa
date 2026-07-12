def format_chat_reply(stored_episodic: list, stored_semantic: list, retrieved_text: str = None) -> str:
    parts = []

    if stored_episodic or stored_semantic:
        parts.append("Got it, I've noted that down.")

    if retrieved_text:
        parts.append(retrieved_text)

    if not parts:
        return "I didn't find anything to save or answer in that message."

    return " ".join(parts)


def format_episodic_result(results) -> str:
    if not results:
        return "We haven't discussed anything related to this yet."
    lines = []
    for item in results:
        when = item.event_timestamp.strftime("%b %d, %Y") if item.event_timestamp else ""
        lines.append(f"- {item.content}" + (f" ({when})" if when else ""))
    return "Here's what I found:\n" + "\n".join(lines)


def format_semantic_result(results) -> str:
    if not results:
        return "We haven't discussed anything related to this yet."
    values = [r["value"] for r in results]
    return "Here's what I found: " + ", ".join(values)