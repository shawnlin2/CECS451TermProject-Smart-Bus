import pandas as pd
import numpy as np

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
def load_and_filter(csv_path: str = 'public_transport_delays.csv'):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not find '{csv_path}'. Run from project root or provide correct path.") from exc

    exclude_columns = [
        'trip_id', 'date', 'route_id', 'origin_station', 'destination_station',
        'scheduled_departure', 'scheduled_arrival', 'actual_departure_delay_min',
        'event_type', 'event_attendance_est'
    ]

    df_cleaned = df.drop(columns=exclude_columns, errors='ignore')

    if 'transport_type' not in df_cleaned.columns:
        raise KeyError("Required column 'transport_type' not found in data.")

    mask = df_cleaned['transport_type'].astype(str).str.strip().str.lower() == 'bus'
    df_cleaned = df_cleaned.loc[mask].copy()

    if df_cleaned.shape[0] == 0:
        raise ValueError("No rows left after filtering for transport_type == 'bus'.")

    if 'delayed' not in df_cleaned.columns:
        raise KeyError("Required target column 'delayed' not found in data after filtering.")

    # Ensure excluded columns are removed even if they were re-created or had different
    # whitespace/casing after earlier processing. Use errors='ignore' so missing names
    # don't raise an exception.
    df_cleaned = df_cleaned.drop(columns=exclude_columns, errors='ignore')

    return df_cleaned


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
    

    # Use actual arrival delay (minutes) as the target variable
    if 'actual_arrival_delay_min' not in df_cleaned.columns:
        raise KeyError("Required target column 'actual_arrival_delay_min' not found in data after filtering.")

    # Ensure numeric target and replace missing with 0.0 (or consider dropping NaNs)
    y = pd.to_numeric(df_cleaned['actual_arrival_delay_min'], errors='coerce').fillna(0.0)
    X = df_cleaned.drop(columns=['actual_arrival_delay_min'])
    print(y.describe())

    # Preprocess features: parse datetimes, encode categoricals, ensure numeric-only
    X_proc = X.copy()

    # Parse any object columns that look like datetimes
    for col in list(X_proc.columns):
        if X_proc[col].dtype == object or pd.api.types.is_string_dtype(X_proc[col]):
            parsed = pd.to_datetime(X_proc[col], errors='coerce', infer_datetime_format=True)
            # If parsing yields some dates, extract components and drop original
            if parsed.notna().any():
                X_proc[col + '_year'] = parsed.dt.year
                X_proc[col + '_month'] = parsed.dt.month
                X_proc[col + '_day'] = parsed.dt.day
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

    # Ensure explicit, exact-name leakage columns are removed as well
    exact_leak_cols = ['actual_arrival_delay_min', 'actual_departure_delay_min']
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

    # Split data into train/test using stratify for classification when possible
    stratify = y if is_classification_target(y) else None
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    # Build preprocessing + SMOTE + classifier pipeline for classification
    if is_classification_target(y):
        # Use a RepeatedStratifiedKFold for stable CV estimates
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

        # Pipeline keeps SMOTE in-place (we will tune SMOTE + classifier together)
        pipeline = ImbPipeline(steps=[
            ('preproc', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])

        # Focus on macro F1 for overall class balance while allowing
        # later custom searches for minority recall if needed.
        scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}
        refit_metric = 'f1_macro'

        # Expanded hyperparameter space including SMOTE sampling strategy
        # and k_neighbors so we can better explore resampling behavior.
        param_dist = {
            'smote__sampling_strategy': [0.5, 0.75, 1.0, 'auto'],
            'smote__k_neighbors': [1, 3, 5, 7],
            'clf__n_estimators': [100, 140, 200, 300],
            'clf__min_samples_leaf': randint(1, 6),
            'clf__max_depth': [None, 5, 8, 12],
            'clf__max_features': ['sqrt', 'log2', 0.5, None],
            'clf__class_weight': [None, 'balanced']
        }

        # Use a larger randomized search to better explore SMOTE+RF interactions
        # (n_iter chosen to balance exploration and runtime). Results will be
        # saved to `smote_extended_search_results.json` for later inspection.
        try:
            import json
            n_iter_search = 60
            rs = RandomizedSearchCV(pipeline, param_dist,
                                    n_iter=n_iter_search, cv=cv,
                                    scoring=scoring, refit=refit_metric,
                                    n_jobs=-1, random_state=42, verbose=2)
            rs.fit(train_x, train_y)
            print("Best params from extended SMOTE RandomizedSearchCV:", rs.best_params_)
            best_pipeline = rs.best_estimator_

            # Save summarized search output (best params, best score, top candidates)
            out = {
                'best_params': rs.best_params_,
                'best_score': float(rs.best_score_),
                'n_iter': n_iter_search
            }
            try:
                # attempt to store the top 10 candidates by mean test score
                cvres = rs.cv_results_
                order = np.argsort(cvres['mean_test_score'])[::-1][:10]
                top = []
                for i in order:
                    top.append({'mean_test_score': float(cvres['mean_test_score'][i]), 'params': cvres['params'][i]})
                out['top_candidates'] = top
            except Exception:
                pass

            with open('smote_extended_search_results.json', 'w') as fh:
                json.dump(out, fh, indent=2)
        except Exception as e:
            print("Extended RandomizedSearchCV failed or skipped:", e)
            # Fallback: fit a default SMOTE pipeline on training data
            pipeline.fit(train_x, train_y)
            best_pipeline = pipeline
    else:
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

        pipeline = ImbPipeline(steps=[
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

    # Optionally run randomized search on the training set to find better params
    try:
        rs = RandomizedSearchCV(pipeline, param_dist,
                                n_iter=20, cv=cv,
                                scoring=scoring,
                                refit=refit_metric,
                                n_jobs=-1, random_state=42)
        rs.fit(train_x, train_y)
        print("Best params from RandomizedSearchCV:", rs.best_params_)
        best_pipeline = rs.best_estimator_
    except Exception as e:
        print("RandomizedSearchCV failed or skipped:", e)
        # fallback: use default pipeline and fit on full training data
        pipeline.fit(train_x, train_y)
        best_pipeline = pipeline

    # Evaluate on holdout test set
    preds = best_pipeline.predict(test_x)



    if is_classification_target(y):
        acc = accuracy_score(test_y, preds)
        f1 = f1_score(test_y, preds, average='macro')
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1 (macro): {f1:.4f}")
        print("Classification report:\n", classification_report(test_y, preds))
        print("Confusion matrix:\n", confusion_matrix(test_y, preds))
    else:
        mae = mean_absolute_error(test_y, preds)
        r2 = r2_score(test_y, preds)
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R^2: {r2:.4f}")


if __name__ == '__main__':
    run_predicter()

