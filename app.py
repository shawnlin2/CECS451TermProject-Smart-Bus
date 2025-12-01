import os
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import psycopg2
from joblib import load
from inputParse import parse_user_query

load_dotenv()

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "inference_pipeline.joblib")
DATABASE_URL = os.getenv("DATABASE_URL")

model_loaded = False
model_error = None
try:
    if os.path.exists(MODEL_PATH):
        load(MODEL_PATH)
        model_loaded = True
    else:
        model_error = "Model file not found."
except Exception as e:
    model_error = f"Model load error: {e}"


def predict_delay_for_route(route_id):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    rid = str(route_id)
    if "-" in rid:
        cur.execute("SELECT AVG(delay_sec) FROM training_events WHERE route_id = %s;", (rid,))
    else:
        cur.execute(
            "SELECT AVG(delay_sec) FROM training_events WHERE route_id LIKE %s;",
            (f"{rid}-%",),
        )

    avg_sec = cur.fetchone()[0]
    cur.close()
    conn.close()

    if avg_sec is None:
        return None
    return avg_sec / 60.0


def get_route_path(route_id):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    rid = str(route_id)
    if "-" in rid:
        cur.execute(
            """
            SELECT stop_id, MIN(stop_sequence) AS seq
            FROM training_events
            WHERE route_id = %s
            GROUP BY stop_id
            ORDER BY seq;
            """,
            (rid,),
        )
    else:
        cur.execute(
            """
            SELECT stop_id, MIN(stop_sequence) AS seq
            FROM training_events
            WHERE route_id LIKE %s
            GROUP BY stop_id
            ORDER BY seq;
            """,
            (f"{rid}-%",),
        )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [str(r[0]) for r in rows]


def parse_user_time(t):
    for fmt in ("%I %p", "%I:%M %p"):
        try:
            return datetime.strptime(t.upper(), fmt)
        except ValueError:
            continue
    return None


@app.route("/")
def index():
    return render_template("index.html", model_loaded=model_loaded, model_error=model_error)


@app.route("/api/query", methods=["POST"])
def api_query():
    text = (request.json or {}).get("query", "").strip()
    if not text:
        return jsonify({"error": "Empty query"}), 400

    parsed = parse_user_query(text)
    notes = []
    prediction_text = None

    lower = text.lower()
    has_bus_word = any(w in lower for w in ["bus", "line", "route"])
    if parsed.get("bus_number") and not has_bus_word:
        if not parsed.get("time"):
            parsed["time"] = str(parsed["bus_number"])
        parsed["bus_number"] = None

    bus = parsed.get("bus_number")
    user_time_str = parsed.get("time")
    destination = parsed.get("destination")

    if not bus:
        notes.append("No bus number found in query.")
        return jsonify(
            {
                "parsed": parsed,
                "prediction_text": "No prediction",
                "notes": notes,
                "display_stops": [],
                "full_stop_count": 0,
                "path_summary": None,
            }
        )

    delay_min = predict_delay_for_route(bus)
    if delay_min is None:
        notes.append("No historical delay data for this route.")
        delay_text = None
    else:
        delay_abs = abs(delay_min)
        if delay_min < 0:
            delay_text = f"Bus is usually early by {delay_abs:.1f} minutes."
        else:
            delay_text = f"Estimated delay: {delay_min:.1f} minutes."

    buffer_min = 5
    effective_delay = (abs(delay_min) if delay_min is not None else 0.0) + buffer_min

    if delay_text:
        if user_time_str:
            target = parse_user_time(user_time_str)
            if target:
                today = datetime.now()
                target = target.replace(year=today.year, month=today.month, day=today.day)
                depart_time = target - timedelta(minutes=effective_delay)
                depart_str = depart_time.strftime("%I:%M %p").lstrip("0")

                parsed["time"] = user_time_str
                prediction_text = (
                    f"{delay_text} To arrive by {user_time_str}, "
                    f"you should leave around {depart_str}."
                )
            else:
                prediction_text = delay_text
        else:
            eta = datetime.now() + timedelta(minutes=effective_delay)
            eta_str = eta.strftime("%I:%M %p").lstrip("0")
            parsed["time"] = eta_str
            prediction_text = f"{delay_text} Expected arrival around {eta_str}."
    else:
        prediction_text = "No prediction"

    display_stops = []
    full_count = 0
    path_summary = None

    try:
        all_stops = get_route_path(bus)
        full_count = len(all_stops)
        if full_count > 0:
            if full_count <= 10:
                display_stops = all_stops
            else:
                display_stops = all_stops[:5] + all_stops[-5:]

            if destination:
                path_summary = (
                    f"Path toward {destination} on route {bus} — "
                    f"showing {len(display_stops)} of {full_count} stops."
                )
            else:
                path_summary = (
                    f"Showing {len(display_stops)} of {full_count} stops on route {bus}."
                )
        else:
            notes.append("No path data for this route.")
    except Exception as e:
        notes.append(f"Path error: {e}")

    return jsonify(
        {
            "parsed": parsed,
            "prediction_text": prediction_text,
            "notes": notes,
            "display_stops": display_stops,
            "full_stop_count": full_count,
            "path_summary": path_summary,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
