import pandas as pd
import numpy as np
import time
import warnings
import os
from joblib import dump
import json
from sklearn.model_selection import cross_validate, RepeatedKFold, RandomizedSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from scipy.stats import randint
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import sys

#Grabs and formats the dataset
def load_and_filter(csv_path: str = 'data/training_events.csv', db_uri: str = None, table_name: str = 'public_transport_delays'):
    """Load dataset either from a CSV file (default) or from a database.

    Parameters
    - csv_path: local CSV file path used when `db_uri` is not provided.
    - db_uri: optional SQLAlchemy-style database URI (e.g. 'sqlite:///data.db' or 'postgresql://...').
    - table_name: table name to read when using `db_uri`.

    Returns a cleaned DataFrame filtered to buses and with required columns validated.
    """
    df = None
    if db_uri:
        # Try to load via SQLAlchemy/pandas first (supports many DBs)
        try:
            from sqlalchemy import create_engine
            engine = create_engine(db_uri)
            # Use pandas read_sql_table when available (works with SQLAlchemy engines)
            try:
                df = pd.read_sql_table(table_name, con=engine)
            except Exception:
                # Fallback to read_sql_query in case read_sql_table is not supported for this backend
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", con=engine)
        except Exception:
            # As a last resort support basic sqlite URI using the stdlib sqlite3 module
            try:
                import sqlite3
                # handle URIs like sqlite:///absolute/path or sqlite:///:memory:
                if db_uri.startswith('sqlite:///'):
                    db_path = db_uri.split('sqlite:///', 1)[1]
                elif db_uri.startswith('sqlite://'):
                    db_path = db_uri.split('sqlite://', 1)[1]
                else:
                    raise
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                conn.close()
            except Exception as exc:
                raise RuntimeError(f"Failed to load table '{table_name}' from database URI '{db_uri}': {exc}") from exc
    else:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Could not find '{csv_path}'. Run from project root or provide correct path.") from exc

    # Prefer `delay_sec` as the target. If only `delay_min` is present,
    # convert minutes -> seconds (rounded) to reduce numeric cardinality.
    print(df.columns)
    if 'delay_sec' not in df.columns:
        if 'delay_min' in df.columns:
            df['delay_sec'] = (pd.to_numeric(df['delay_min'], errors='coerce') * 60).round().fillna(0).astype(int)
        else:
            raise KeyError("Required target column 'delay_sec' or 'delay_min' not found in data after loading.")

    # Drop non-informative identifier / snapshot columns when present
    df = df.drop(columns=['id', 'snapshot_utc'], errors='ignore')

    # Return a deterministic sample of up to 5000 rows for quicker iteration.
    if df.shape[0] >= 5000:
        return df.sample(n=5000, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)


def is_classification_target(y: pd.Series) -> bool:
    # Treat as classification if non-numeric or low-cardinality
    if not pd.api.types.is_numeric_dtype(y):
        return True
    # numeric but small number of unique values -> classification
    return y.nunique() <= 10


def summarize(scores: dict):
    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}


def run_predicter():
    try:
        df_cleaned = load_and_filter()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
    

    # Feature engineering: scheduled arrival hour/day, departure delay, route historical avg
    # Parse scheduled arrival datetimes where possible (CSV uses `scheduled_arrival_dt`)
    if 'scheduled_arrival_dt' in df_cleaned.columns:
        sched_arr = pd.to_datetime(df_cleaned['scheduled_arrival_dt'], errors='coerce', infer_datetime_format=True)
        df_cleaned['sched_arr_hour'] = sched_arr.dt.hour
        df_cleaned['sched_arr_dow'] = sched_arr.dt.dayofweek
    else:
        df_cleaned['sched_arr_hour'] = np.nan
        df_cleaned['sched_arr_dow'] = np.nan

    # Include observed departure delay if available (numeric). CSV doesn't include a departure delay column,
    # so set a default if missing.
    if 'actual_departure_delay_min' in df_cleaned.columns:
        df_cleaned['actual_departure_delay_min'] = pd.to_numeric(df_cleaned['actual_departure_delay_min'], errors='coerce')
    else:
        df_cleaned['actual_departure_delay_min'] = 0.0

    # Route-level historical average delay (helps regularize per-route behavior); safe if route_id present
    if 'route_id' in df_cleaned.columns and 'delay_sec' in df_cleaned.columns:
        df_cleaned['route_avg_delay'] = df_cleaned.groupby('route_id')['delay_sec'].transform('mean')
    else:
        df_cleaned['route_avg_delay'] = np.nan

    # Use `delay_sec` (seconds) as the target. If the column isn't present here
    # something went wrong because `load_and_filter` guarantees one of them.
    if 'delay_sec' not in df_cleaned.columns:
        raise KeyError("Required target column 'delay_sec' not found in data after loading.")

    y = pd.to_numeric(df_cleaned['delay_sec'], errors='coerce').fillna(0.0)
    X = df_cleaned.drop(columns=['delay_sec'])
    print(y.describe())

    # Preprocess features: parse datetimes, encode categoricals, ensure numeric-only
    X_proc = X.copy()

    # Parse any object columns that look like datetimes
    for col in list(X_proc.columns):
        if X_proc[col].dtype == object or pd.api.types.is_string_dtype(X_proc[col]):
            parsed = pd.to_datetime(X_proc[col], errors='coerce', infer_datetime_format=True)
            # If parsing yields some dates, extract time-of-day and weekday only (avoid calendar/date fields)
            if parsed.notna().any():
                X_proc[col + '_dow'] = parsed.dt.dayofweek
                X_proc[col + '_hour'] = parsed.dt.hour
                X_proc.drop(columns=[col], inplace=True)

    # At this point we have numeric features and some remaining object/categorical
    # columns (e.g., weather, season). We'll move categorical encoding into the
    # model pipeline using a ColumnTransformer + OneHotEncoder to avoid leaking
    # information during cross-validation.
    # Identify numeric vs categorical columns (after datetime expansion)
    numeric_cols = X_proc.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_proc.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Aggressively remove leakage-like columns that may have been recreated
    leakage_patterns = ['actual_arrival', 'actual_departure']
    cols_to_drop = [c for c in X_proc.columns if any(p.lower() in c.lower() for p in leakage_patterns)]
    if cols_to_drop:
        print("Dropping leakage columns:", cols_to_drop)
        X_proc = X_proc.drop(columns=cols_to_drop, errors='ignore')

    # Ensure explicit, exact-name leakage columns are removed as well (drop delay columns and seconds)
    exact_leak_cols = ['delay_min', 'delay_sec', 'actual_arrival_delay_min', 'actual_departure_delay_min']
    present_exact = [c for c in exact_leak_cols if c in X_proc.columns]
    if present_exact:
        print('Dropping exact leakage columns:', present_exact)
        X_proc = X_proc.drop(columns=present_exact, errors='ignore')

    # Final sanity check: assert leakage cols not present
    leftover = [c for c in exact_leak_cols if c in X_proc.columns]
    if leftover:
        raise RuntimeError(f"Leakage columns still present after cleaning: {leftover}")

    # Recompute numeric/categorical lists after any leakage drop
    numeric_cols = X_proc.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_proc.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Build transformers: numeric pipeline and categorical pipeline
    numeric_transformer = SkPipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())])
    # Instantiate OneHotEncoder in a way that's compatible with multiple
    # scikit-learn versions (older versions use `sparse`, newer use `sparse_output`).
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

    categorical_transformer = SkPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    X = X_proc

    print(f"Data prepared: {X.shape[0]} rows, {X.shape[1]} features")

    # Split data into train / valid / test. Use `test` as final holdout,
    # and `valid` to evaluate RandomizedSearchCV candidates.
    stratify = y if is_classification_target(y) else None
    X_temp, test_x, y_temp, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    stratify2 = y_temp if is_classification_target(y_temp) else None
    # make valid 25% of the remaining (0.25 * 0.8 = 0.2 overall)
    train_x, valid_x, train_y, valid_y = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=stratify2)

    # Build preprocessing + SMOTE + classifier pipeline for classification
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

    pipeline = SkPipeline(steps=[
        ('preproc', preprocessor),
        ('clf', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    scoring = 'neg_mean_absolute_error'
    refit_metric = True

    param_dist = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [5, 8, 12, None],
        'clf__min_samples_leaf': randint(5, 10),
        'clf__max_features': ['sqrt', 'log2', 0.5, None]
    }

    try:
        rs = RandomizedSearchCV(pipeline, param_dist,
                                n_iter=20, cv=cv,
                                scoring=scoring,
                                refit=refit_metric,
                                n_jobs=-1, random_state=42)
        rs_success = False
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Found unknown categories.*", category=UserWarning)
                rs.fit(train_x, train_y)
            rs_success = True
            print("Best params from RandomizedSearchCV:", rs.best_params_)
            best_pipeline = rs.best_estimator_
        except Exception as e:
            print("RandomizedSearchCV fit failed:", e)
            rs_success = False
            raise
    except Exception as e:
        print("RandomizedSearchCV failed or skipped:", e)
        # fallback: use default pipeline and fit on full training data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*", category=UserWarning)
            pipeline.fit(train_x, train_y)
        best_pipeline = pipeline
        rs_success = False

    # Evaluate the selected estimator on the validation set to check RandomizedSearchCV
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*", category=UserWarning)
            val_preds = best_pipeline.predict(valid_x)

        
        val_mae = mean_absolute_error(valid_y, val_preds)
        val_r2 = r2_score(valid_y, val_preds)
        print(f"Validation MAE: {val_mae:.4f} Seconds")
        print(f"Validation R^2: {val_r2:.4f}")

        # Refit the chosen pipeline on train + valid for final evaluation on test
        combined_x = pd.concat([train_x, valid_x], ignore_index=True)
        combined_y = pd.concat([train_y, valid_y], ignore_index=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories.*", category=UserWarning)
            best_pipeline.fit(combined_x, combined_y)
        print("Refit best pipeline on train+valid for final test evaluation.")
    except Exception as e:
        print("Warning: validation evaluation or refit failed:", e)

    # Evaluate on holdout test set
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories.*", category=UserWarning)
        preds = best_pipeline.predict(test_x)




    mae = mean_absolute_error(test_y, preds)
    r2 = r2_score(test_y, preds)
    print(f"Test MAE: {mae:.4f} seconds")
    print(f"Test R^2: {r2:.4f}")

    # Save an inference pipeline (preprocessor + estimator) for later use.
    try:
        preproc = best_pipeline.named_steps.get('preproc') if hasattr(best_pipeline, 'named_steps') else None
        clf = best_pipeline.named_steps.get('clf') if hasattr(best_pipeline, 'named_steps') else None
        if preproc is not None and clf is not None:
            inference_pipe = SkPipeline([('preproc', preproc), ('clf', clf)])
        else:
            # Fallback: save the whole fitted pipeline/estimator
            inference_pipe = best_pipeline

        os.makedirs('models', exist_ok=True)
        # Save inference pipeline (preproc + clf) for lightweight serving
        dump(inference_pipe, os.path.join('models', 'inference_pipeline.joblib'))
        print("Saved inference pipeline to models/inference_pipeline.joblib")

        # Also save the full fitted pipeline (includes any resampling steps)
        try:
            dump(best_pipeline, os.path.join('models', 'best_pipeline_full.joblib'))
            print("Saved full trained pipeline to models/best_pipeline_full.joblib")
        except Exception as e:
            print("Warning: failed to save full pipeline:", e)

        # Persist chosen classifier parameters for inspection
        try:
            # Prefer to save the RandomizedSearchCV best params if available
            params_out = {'source': None, 'best_params': None}
            if 'rs' in locals() and isinstance(rs, RandomizedSearchCV) and rs_success:
                params_out['source'] = 'randomized_search'
                params_out['best_params'] = rs.best_params_
            else:
                params_out['source'] = 'fallback_pipeline'
                if hasattr(best_pipeline, 'named_steps') and best_pipeline.named_steps.get('clf') is not None:
                    params_out['best_params'] = best_pipeline.named_steps['clf'].get_params()
                else:
                    params_out['best_params'] = best_pipeline.get_params()

            with open(os.path.join('models', 'best_params.json'), 'w') as fh:
                json.dump(params_out, fh, indent=2, default=str)
            print('Saved best parameters to models/best_params.json (source=%s)' % params_out['source'])
        except Exception as e:
            print('Warning: failed to save best params:', e)
    except Exception as e:
        print("Warning: failed to save inference pipeline:", e)


if __name__ == '__main__':
    start_time = time.time()
    run_predicter()
    print((time.time() - start_time))

