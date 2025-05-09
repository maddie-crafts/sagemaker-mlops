import sys
import subprocess

# --------- Install Packages if Needed ---------
def install_packages(packages):
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            sys.exit(1)

REQUIRED_PACKAGES = [
    "boto3", "spacy==3.7.0", 
    "psycopg2-binary", "sagemaker", "numpy"
]
install_packages(REQUIRED_PACKAGES)

import json
import time
import logging
import argparse
import datetime
import boto3
import spacy
import psycopg2
import pandas as pd
import numpy as np
from spacy.tokens import DocBin

# --------- Setup Logging ---------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_db_credentials(secret_name) -> dict:
    client = boto3.client("secretsmanager")
    secret = json.loads(client.get_secret_value(SecretId=secret_name)["SecretString"])
    return {
        "host": secret["host"],
        "port": secret["port"],
        "dbname": secret["dbname"],
        "user": secret["username"],
        "password": secret["password"]
    }

def create_connection(credentials: dict, max_attempts: int = 5):
    for attempt in range(max_attempts):
        try:
            return psycopg2.connect(
                host=credentials["host"],
                port=credentials["port"],
                database=credentials["dbname"],
                user=credentials["user"],
                password=credentials["password"]
            )
        except psycopg2.OperationalError as e:
            logger.warning(f"Connection attempt {attempt+1} failed: {e}")
            time.sleep((attempt + 1) ** 2)
    raise ConnectionError("Failed to connect to database after retries.")

def fetch_coded_data(credentials: dict, table: str) -> list:
    conn = create_connection(credentials)
    SQL = f"""
        SELECT unique_id, text, start_char, end_char, span, span_text, scored_date FROM {table};
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(SQL)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return []
    finally:
        conn.close()

def transform_data_for_training(data: list, span_theme: str) -> list:
    transformed = []
    for record in data:
        text = record.get("text")
        if not text:
            continue

        spans = []
        # Match on the 'span_theme' field, not dynamic span_theme key
        if record.get('span') == span_theme:
            spans.append({
                "start": record["start_char"],
                "end": record["end_char"],
                "label": span_theme
            })
        
        if spans:
            transformed.append({"text": text, "spans": spans})
    return transformed


def convert_to_docbin(nlp, examples: list) -> DocBin:
    doc_bin = DocBin()
    for example in examples:
        text = example["text"]
        doc = nlp.make_doc(text)
        spans = example["spans"]

        # Remove overlaps
        spans.sort(key=lambda span: (span["start"], - span["end"]))
        non_overlapping_spans = []
        for span in spans:
            if non_overlapping_spans and non_overlapping_spans[-1]["end"] > span["start"]:
                continue
            non_overlapping_spans.append(span)

        ents = []
        for span_info in non_overlapping_spans:
            span = doc.char_span(span_info["start"], span_info["end"], label=span_info["label"])
            if span:
                ents.append(span)
        
        doc.spans["sc"] = ents
        doc_bin.add(doc)
    return doc_bin

# --------- Main Process ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--span_theme", type=str, default="organization")
    parser.add_argument("--scored-date-greater-than", default="2025-01-01", type=str)
    parser.add_argument("--secret_name", default="database-secrets-in-secrets-manager", type=str)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    span_theme = args.span_theme
    scored_after_date = args.scored_date_greater_than

    logger.info(f"Starting data processing for span theme: {span_theme}")

    creds = get_db_credentials(args.secret_name)
    manually_coded = fetch_coded_data(creds, table='training_data')

    if not manually_coded:
        logger.error("No data retrieved. Exiting...")
        sys.exit(1)

    try:
        cutoff_date = datetime.datetime.strptime(scored_after_date, "%Y-%m-%d").date()
        logger.info(f"Using cutoff date: {cutoff_date}")
    except ValueError:
        logger.error(f"Invalid date format for --scored-date-greater-than: {scored_after_date}. Expected format: YYYY-MM-DD.")
        sys.exit(1)

    np.random.seed(42)
    np.random.shuffle(manually_coded)

    total = len(manually_coded)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    train_data = transform_data_for_training(manually_coded[:train_end], span_theme)
    val_data = transform_data_for_training(manually_coded[train_end:val_end], span_theme)
    test_data = manually_coded[val_end:]
    
    # Save Test Data
    test_df = pd.DataFrame(test_data)
    test_df = test_df[test_df['span'].str.contains(span_theme)]
    test_df.to_csv(f"{base_dir}/test/test.csv", index=False)
    logger.info(f"Saved test data ({len(test_df)} examples).")

    # Save Train and Validation DocBin
    nlp = spacy.blank("en")
    doc_bin_train = convert_to_docbin(nlp, train_data)
    doc_bin_val = convert_to_docbin(nlp, val_data)

    doc_bin_train.to_disk(f"{base_dir}/train/train.spacy")
    doc_bin_val.to_disk(f"{base_dir}/validation/validation.spacy")

    logger.info(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")
    logger.info("Data processing completed successfully.")

if __name__ == "__main__":
    main()