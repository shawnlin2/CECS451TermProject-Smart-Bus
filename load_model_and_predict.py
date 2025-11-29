"""Simple example: load the saved inference pipeline and predict on a CSV.

Usage:
  python load_model_and_predict.py --model models/inference_pipeline.joblib --input data/la_metro_training_snapshot.csv --n 10

Notes:
- This script assumes `models/inference_pipeline.joblib` exists (saved by `delay_prediction.py`).
- If the saved preprocessor expects preprocessed columns (e.g. datetime expansions), you may need to apply the same preprocessing as in `run_predicter()` before calling `.predict()`.
"""

import argparse
import os
import sys
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from joblib import load


def main():
    p = argparse.ArgumentParser(description='Load saved inference pipeline and predict on CSV rows')
    p.add_argument('--model', '-m', default='models/inference_pipeline.joblib', help='Path to saved joblib model')
    p.add_argument('--input', '-i', default='data/training_events.csv', help='CSV input file with raw rows')
    p.add_argument('--n', '-n', type=int, default=10, help='Number of rows to predict (head)')
    p.add_argument('--output', '-o', default='models/predictions.csv', help='Output CSV path')
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    model_built_from_params = False
    model = None
    # Try loading the requested model; if missing, try full pipeline; otherwise build from params
    # Try loading the requested model if it exists
    if os.path.exists(args.model):
        try:
            model = load(args.model)
        except Exception as e:
            print(f"Warning: failed to load requested model '{args.model}': {e}", file=sys.stderr)
            model = None

    # If no model loaded, prefer building from params (backup) instead of .joblib
    if model is None:
        params_path = os.path.join('models', 'best_params.json')
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r') as fh:
                    params_out = json.load(fh)
                best_params = params_out.get('best_params', {})
                # if keys are prefixed like 'clf__param', strip prefix
                est_params = {}
                for k, v in best_params.items():
                    if k.startswith('clf__'):
                        est_params[k.replace('clf__', '')] = v
                    else:
                        est_params[k] = v
                # Build a RandomForestRegressor with these params as a fallback
                print(f"Building RandomForestRegressor from params: {list(est_params.keys())}")
                model = RandomForestRegressor(**est_params)
                # note: the estimator is not fitted — we'll refuse to fit here unless requested
                model_built_from_params = True
            except Exception as e3:
                print(f"Warning: failed to build estimator from params: {e3}", file=sys.stderr)
                model = None
        else:
            # as last resort, try full pipeline joblib
            fallback_full = os.path.join('models', 'best_pipeline_full.joblib')
            if os.path.exists(fallback_full):
                try:
                    model = load(fallback_full)
                    print(f"Loaded fallback full pipeline from {fallback_full}")
                except Exception as e2:
                    print(f"Warning: failed to load fallback full pipeline: {e2}", file=sys.stderr)
                    model = None
    # If still None, exit
    if model is None:
        print("No usable model found. Please provide a valid model file or create models/best_pipeline_full.joblib or models/best_params.json", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.input)

    # Drop known target/id/snapshot columns if present so we pass only features
    drop_cols = ['delay_sec', 'delay_min', 'id', 'snapshot_utc']
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')

    df_in = df.head(args.n).copy()

    # Preprocessing to match training-time feature engineering
    # 1) If `scheduled_arrival_dt` exists, create `sched_arr_hour` and `sched_arr_dow`
    if 'scheduled_arrival_dt' in df_in.columns:
        sched_arr = pd.to_datetime(df_in['scheduled_arrival_dt'], errors='coerce', infer_datetime_format=True)
        df_in['sched_arr_hour'] = sched_arr.dt.hour
        df_in['sched_arr_dow'] = sched_arr.dt.dayofweek

    # 2) For any object/string columns that parse as datetimes, expand into components
    for col in list(df_in.columns):
        if df_in[col].dtype == object or pd.api.types.is_string_dtype(df_in[col]):
            parsed = pd.to_datetime(df_in[col], errors='coerce', infer_datetime_format=True)
            if parsed.notna().any():
                df_in[col + '_year'] = parsed.dt.year
                df_in[col + '_month'] = parsed.dt.month
                df_in[col + '_day'] = parsed.dt.day
                df_in[col + '_dow'] = parsed.dt.dayofweek
                df_in[col + '_hour'] = parsed.dt.hour
                df_in.drop(columns=[col], inplace=True)

    # 3) Ensure `actual_departure_delay_min` exists (training set defaulted to 0.0)
    if 'actual_departure_delay_min' in df_in.columns:
        df_in['actual_departure_delay_min'] = pd.to_numeric(df_in['actual_departure_delay_min'], errors='coerce')
    else:
        df_in['actual_departure_delay_min'] = 0.0

    # 4) route_avg_delay: compute from provided rows when possible, otherwise leave NaN
    if 'route_id' in df_in.columns and 'delay_sec' in df.columns:
        # if input included historical delays, compute per-route mean
        temp = df[[c for c in df.columns if c in ['route_id', 'delay_sec']]]
        df_in['route_avg_delay'] = df_in['route_id'].map(temp.groupby('route_id')['delay_sec'].mean())
    else:
        df_in['route_avg_delay'] = np.nan

    # Predict: if model was built from params only (unfitted), refuse to predict — user should supply a fitted pipeline.
    if model_built_from_params:
        print("Model was constructed from params but is not fitted. Provide a fitted pipeline (.joblib) to predict or run training to fit the estimator.", file=sys.stderr)
        sys.exit(3)

    try:
        preds = model.predict(df_in)
    except Exception as e:
        # Try a basic fallback: one-hot encode then predict
        try:
            df_enc = pd.get_dummies(df_in)
            preds = model.predict(df_enc)
        except Exception as e2:
            print("Model prediction failed:", e2, file=sys.stderr)
            print("Hint: the saved pipeline's preprocessor may expect additional preprocessed columns.", file=sys.stderr)
            print("If so, re-run training code or provide a fitted pipeline. Alternatively, create a fitted estimator.", file=sys.stderr)
            sys.exit(3)

    # Save only the predicted delay to the output CSV. Include seconds and minutes.
    out = pd.DataFrame({'pred_delay_sec': preds})
    out['pred_delay_min'] = out['pred_delay_sec'] / 60.0

    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} predicted delays to {args.output}")


if __name__ == '__main__':
    main()
