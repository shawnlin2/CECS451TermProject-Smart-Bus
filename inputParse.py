import re
import spacy

nlp = spacy.load("en_core_web_sm")

DEST_KEYWORDS = {
    "downtown": "Downtown",
    "long beach": "Long Beach",
    "santa monica": "Santa Monica",
    "usc": "USC",
    "union station": "Union Station",
}


def parse_user_query(query: str):
    doc = nlp(query)
    text = query.lower()

    result = {
        "intent": None,
        "bus_number": None,
        "destination": None,
        "time": None,
    }

    # basic intent
    if "next" in text and "bus" in text:
        result["intent"] = "get_next_bus"
    elif "leave" in text or "when should i" in text:
        result["intent"] = "plan_departure"
    elif "delay" in text or "late" in text:
        result["intent"] = "check_delay"
    else:
        result["intent"] = "unknown"

    # bus number: "bus 70", "route 232", "take 720"
    m = re.search(r"\b(?:bus|line|route)?\s*(\d{1,4})\b", text)
    if m:
        result["bus_number"] = int(m.group(1))

    # time from entities
    for ent in doc.ents:
        if ent.label_ in ("TIME", "DATE"):
            result["time"] = ent.text
            break

    # destination from entities
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "FAC", "FACILITY"):
            result["destination"] = ent.text
            break

    # fallback: keyword destinations like "downtown"
    if not result["destination"]:
        for key, pretty in DEST_KEYWORDS.items():
            if key in text:
                result["destination"] = pretty
                break

    return result
