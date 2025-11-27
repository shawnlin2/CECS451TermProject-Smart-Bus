import io
import math
import zipfile
from datetime import datetime, timedelta, timezone
import pytz

from functools import lru_cache

import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

import glob
from pathlib import Path

load_dotenv()
#Connect to Database
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Static GTFS (LA Metro bus)
GTFS_STATIC_URL = "https://gitlab.com/LACMTA/gtfs_bus/raw/master/gtfs_bus.zip"

# Open-Meteo (no key)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


#Get LA time
LA_TZ = pytz.timezone("America/Los_Angeles")

# 1) Load static GTFS (once)

def load_static_gtfs():
    print("Downloading static GTFS...")
    resp = requests.get(GTFS_STATIC_URL, timeout=30)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))

    trips = pd.read_csv(z.open("trips.txt"))
    stop_times = pd.read_csv(z.open("stop_times.txt"))
    stops = pd.read_csv(z.open("stops.txt"))

    stop_times = stop_times[
        ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    ]
    trips = trips[["trip_id", "route_id", "direction_id"]]
    stops = stops[["stop_id", "stop_lat", "stop_lon"]]

    # Normalize ID columns to strings to avoid dtype mismatch when merging with realtime feed
    stop_times["trip_id"] = stop_times["trip_id"].astype(str)
    stop_times["stop_id"] = stop_times["stop_id"].astype(str)
    trips["trip_id"] = trips["trip_id"].astype(str)
    stops["stop_id"] = stops["stop_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)

    static = (
        stop_times
        .merge(trips, on="trip_id", how="left")
        .merge(stops, on="stop_id", how="left")
    )
    return static


STATIC_GTFS = load_static_gtfs()

# 2) Pull GTFS-rt from Transitland

def fetch_gtfsrt_from_file(path):
    """
    Download latest GTFS-rt protobuf for the LA Metro bus realtime feed
    via Transitland's latest_realtime endpoint.
    """
    with open(path, "rb") as f:
        return f.read()
    


def parse_trip_updates(pb_bytes):
    """
    Parse a GTFS-rt FeedMessage containing TripUpdates and return a DataFrame.
    """
    times_has_timestamp = 0
    dont_have_timestamp = 0
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(pb_bytes)

    # ✅ use feed header timestamp if present
    if feed.header.HasField("timestamp"):
        times_has_timestamp += 1
        snapshot_utc = datetime.fromtimestamp(
            feed.header.timestamp,
            tz=timezone.utc
        )
    else:
        dont_have_timestamp += 1
        snapshot_utc = datetime.now(tz=timezone.utc)

    rows = []
    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue

        tu = entity.trip_update
        trip = tu.trip
        trip_id = str(trip.trip_id) if trip.trip_id is not None else ""

        for stu in tu.stop_time_update:
            stop_id = str(stu.stop_id) if stu.stop_id is not None else ""
            arr = stu.arrival
            dep = stu.departure

            rows.append({
                "snapshot_utc": snapshot_utc,
                "trip_id": trip_id,
                "stop_id": stop_id,
                "arrival_delay_sec": arr.delay if arr.HasField("delay") else None,
                "arrival_time_epoch": arr.time if arr.HasField("time") else None,
                "departure_delay_sec": dep.delay if dep.HasField("delay") else None,
                "departure_time_epoch": dep.time if dep.HasField("time") else None,
            })
    print(times_has_timestamp, dont_have_timestamp)
    return pd.DataFrame(rows)


# 3) Open-Meteo weather helpers

def round_coord(val, step=0.1):
    return round(val / step) * step

@lru_cache(maxsize=1024)
def get_weather_for_cell(lat_cell, lon_cell):
    params = {
        "latitude": lat_cell,
        "longitude": lon_cell,
        "current": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,visibility",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    cur = data.get("current", {}) or {}
    return {
        "temp_c": cur.get("temperature_2m"),
        "precip_mm": cur.get("precipitation"),
        "rel_humidity": cur.get("relative_humidity_2m"),
        "wind_speed_ms": cur.get("wind_speed_10m"),
        "visibility_m": cur.get("visibility"),
    }

def get_weather_for_stop(lat, lon):
    lat_cell = round_coord(lat)
    lon_cell = round_coord(lon)
    return get_weather_for_cell(lat_cell, lon_cell)

# 4) Time helpers

def gtfs_time_to_datetime(time_str, service_date):

    h, m, s = map(int, time_str.split(":"))
    day_offset, h = divmod(h, 24)

    # Build local datetime
    dt_local = datetime(
        service_date.year, service_date.month, service_date.day,
        h, m, s
    ) + timedelta(days=day_offset)

    # Attach LA timezone
    dt_local = LA_TZ.localize(dt_local)

    # Convert to UTC (this is what GTFS-rt epoch uses)
    return dt_local.astimezone(pytz.UTC)


def add_time_features(df, col_name="scheduled_arrival_dt"):
    # scheduled times are stored as UTC-aware datetimes; convert to local tz for time-of-day features
    sched = df[col_name]
    # ensure tz-aware in UTC
    try:
        if sched.dt.tz is None:
            sched = sched.dt.tz_localize(pytz.UTC)
        else:
            sched = sched.dt.tz_convert(pytz.UTC)
    except Exception:
        # if any weirdness, assume naive UTC
        sched = pd.to_datetime(sched).dt.tz_localize(pytz.UTC)

    # convert to LA local time
    sched_local = sched.dt.tz_convert(LA_TZ)

    df["hour_of_day"] = sched_local.dt.hour + sched_local.dt.minute / 60.0
    df["day_of_week"] = sched_local.dt.weekday  # Monday=0
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    df["is_rush_hour"] = df["hour_of_day"].between(7, 10) | df["hour_of_day"].between(16, 19)

    df["hour_sin"] = df["hour_of_day"].apply(lambda h: math.sin(2 * math.pi * h / 24))
    df["hour_cos"] = df["hour_of_day"].apply(lambda h: math.cos(2 * math.pi * h / 24))
    return df

# 5) Build training rows

def build_training_rows_from_pb(pb_bytes):
    rt_df = parse_trip_updates(pb_bytes)

    if rt_df.empty:
        print("No realtime rows in this snapshot.")
        return pd.DataFrame()

    # Join realtime with static GTFS on (trip_id, stop_id)
    joined = rt_df.merge(
        STATIC_GTFS,
        on=["trip_id", "stop_id"],
        how="inner",
        suffixes=("", "_static"),
    )


    # Compute time
    def compute_times(row):
        # 1) actual arrival from GTFS-rt epoch (UTC-aware if present)
        actual_utc = None
        if pd.notnull(row["arrival_time_epoch"]):
            actual_utc = datetime.fromtimestamp(int(row["arrival_time_epoch"]),
                                                tz=timezone.utc)

        # 2) pick LA service date from actual time (or snapshot as fallback)
        if actual_utc is not None:
            service_date_local = actual_utc.astimezone(LA_TZ).date()
        else:
            service_date_local = row["snapshot_utc"].astimezone(LA_TZ).date()

        # 3) scheduled arrival string ("HH:MM:SS") -> UTC using that service date
        scheduled_utc = gtfs_time_to_datetime(row["arrival_time"], service_date_local)

        # 4) if actual_utc missing but we have delay, derive it from schedule + delay
        if actual_utc is None and pd.notnull(row["arrival_delay_sec"]):
            actual_utc = scheduled_utc + timedelta(seconds=row["arrival_delay_sec"])

        return pd.Series({
            "scheduled_arrival_dt": scheduled_utc,
            "actual_arrival_dt": actual_utc,
        })

    times = joined.apply(compute_times, axis=1)
    joined = pd.concat([joined, times], axis=1)

    joined["delay_sec"] = (
        joined["actual_arrival_dt"] - joined["scheduled_arrival_dt"]
    ).dt.total_seconds()

    joined["delay_min"] = joined["delay_sec"] / 60.0

    # Filter out rows with no delay
    joined = joined[joined["delay_sec"].notna()].copy()

    # Add time features
    joined = add_time_features(joined, "scheduled_arrival_dt")

    # Add weather (with caching by grid cell)
    weather_cols = ["temp_c", "precip_mm", "rel_humidity", "wind_speed_ms", "visibility_m"]
    for col in weather_cols:
        joined[col] = None

    for idx, row in joined.iterrows():
        lat = row["stop_lat"]
        lon = row["stop_lon"]
        if pd.isna(lat) or pd.isna(lon):
            continue
        w = get_weather_for_stop(lat, lon)
        for col in weather_cols:
            joined.at[idx, col] = w.get(col)

    training_cols = [
        "snapshot_utc",
        "route_id",
        "trip_id",
        "stop_id",
        "stop_sequence",
        "scheduled_arrival_dt",
        "actual_arrival_dt",
        "delay_sec",
        "delay_min",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_rush_hour",
        "hour_sin",
        "hour_cos",
        "temp_c",
        "precip_mm",
        "rel_humidity",
        "wind_speed_ms",
        "visibility_m",
    ]

    training_df = joined[training_cols].copy()
    return training_df

def save_training_rows_to_db(df, engine):
    if df.empty:
        print("No rows to save.")
        return
    df.to_sql("training_events", engine, if_exists="append", index=False)
    print(f"Inserted {len(df)} rows into training_events.")

def process_all_snapshots():
    snap_shot_dir = Path('data/real_time_data')
    snap_shot_dir.mkdir(exist_ok=True)
    pb_files = sorted(snap_shot_dir.glob("*.pb"))
    total_rows = 0
    for pb_path in pb_files:
        pb_bytes = fetch_gtfsrt_from_file(pb_path)
        df = build_training_rows_from_pb(pb_bytes)
        if df.empty:
            continue
    
        save_training_rows_to_db(df, engine)
        total_rows += len(df)
    print(f"Done. Total row: {total_rows}")


if __name__ == "__main__":
    process_all_snapshots()
