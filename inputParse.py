import spacy
import re

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def parse_user_query(query: str):
    doc = nlp(query)

    # Default result structure
    result = {
        "intent": None,
        "bus_number": None,
        "destination": None
    }

    # Simple intent detection (can expand later)
    text = query.lower()
    if "next" in text and "bus" in text:
        result["intent"] = "get_next_bus"
    elif "leave" in text or "when should i " in text:
        result["intent"] = "find_departure_time"
    elif "delay" in text:
        result["intent"] = "check_delay"
    else:
        result["intent"] = "unknown"

    # Extract bus number (e.g., 33 in “bus 33”)
    match = re.search(r"\b(?:bus\s*)?(\d{1,3})\b", query)
    if match:
        result["bus_number"] = match.group(1)

    # Extract possible destination/place names
    # spaCy’s named-entity recognition will tag cities, places, etc.
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FACILITY"]:
            result["destination"] = ent.text
            break  # take the first match
        
    for ent in doc.ents:
        if ent.label_ in ["TIME", "DATE"]:
            result["time"] = ent.text
            break
    return result
print(parse_user_query("When should I take the 70 bus if I want to arrive by noon?"))