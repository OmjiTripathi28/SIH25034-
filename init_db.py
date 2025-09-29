# init_db.py
import sqlite3
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import json

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "internships.csv")
DB_PATH = os.path.join(DATA_DIR, "internships.db")
MODEL_DIR = os.path.join(DATA_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def create_db_from_csv(csv_path, db_path):
    df = pd.read_csv(csv_path)

    # Normalize schema according to your CSV
    # (internship_title, company_name, location, start_date, duration, stipend)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    # add missing columns so rest of pipeline won’t break
    for col in ["description", "skills", "sectors", "education"]:
        if col not in df.columns:
            df[col] = ""

    # normalize skills/sectors into lists
    df["skills_list"] = df["skills"].fillna("").apply(
        lambda s: [t.strip().lower() for t in str(s).split(",") if t.strip()]
    )
    df["sectors_list"] = df["sectors"].fillna("").apply(
        lambda s: [t.strip().lower() for t in str(s).split(",") if t.strip()]
    )

    # create DB
    conn = sqlite3.connect(db_path)
    df_to_store = df.copy()
    df_to_store["skills_json"] = df_to_store["skills_list"].apply(json.dumps)
    df_to_store["sectors_json"] = df_to_store["sectors_list"].apply(json.dumps)
    store_cols = [
        "id",
        "internship_title",
        "company_name",
        "location",
        "start_date",
        "duration",
        "stipend",
        "description",
        "skills_json",
        "sectors_json",
        "education",
    ]
    df_to_store[store_cols].to_sql("internships", conn, if_exists="replace", index=False)
    conn.close()
    print(f"DB created at {db_path} with {len(df)} internships.")
    return df


def build_and_save_model(df, model_dir):
    # Build text column from available fields
    df["text"] = (
        df["internship_title"].fillna("")
        + " "
        + df["company_name"].fillna("")
        + " "
        + df["location"].fillna("")
        + " "
        + df["duration"].fillna("")
        + " "
        + df["stipend"].fillna("")
        + " "
        + df["description"].fillna("")
    )

    print("Sample text rows for TF-IDF:\n", df["text"].head())

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, stop_words=None)
    X_text = tfidf.fit_transform(df["text"].astype(str))
    joblib.dump(tfidf, os.path.join(model_dir, "tfidf.joblib"))
    joblib.dump(X_text, os.path.join(model_dir, "X_text.joblib"))

    # Skills multi-hot
    mlb = MultiLabelBinarizer(sparse_output=True)
    skills_matrix = mlb.fit_transform(df["skills_list"])
    joblib.dump(mlb, os.path.join(model_dir, "skills_mlb.joblib"))
    joblib.dump(skills_matrix, os.path.join(model_dir, "skills_matrix.joblib"))

    # Save metadata
    meta_cols = [
        "id",
        "internship_title",
        "company_name",
        "location",
        "start_date",
        "duration",
        "stipend",
        "description",
        "education",
    ]
    df[meta_cols].to_pickle(os.path.join(model_dir, "meta.pkl"))
    print("Model artifacts saved in", model_dir)


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Put your internships csv at {CSV_PATH}")
    df = create_db_from_csv(CSV_PATH, DB_PATH)
    if "skills_list" not in df.columns:
        df["skills_list"] = df["skills"].fillna("").apply(
            lambda s: [t.strip().lower() for t in str(s).split(",") if t.strip()]
        )
    build_and_save_model(df, MODEL_DIR)
    print("Initialization complete.")
