import os
import sys
import json
import pathlib
import tarfile
import subprocess
import logging
import boto3
import pandas as pd
import string

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# --------- Setup Logging ---------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THRESHOLD = 0.2

# --------- Install Required Packages ---------
def install_packages():
    packages = [
        ["spacy==3.7.0"],
        ["spacy-transformers"], 
        ["thinc==8.2.5"],  
        ["pydantic>=2.1.1"],
        ["wasabi==1.1.3"], 
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkg])

install_packages()

import spacy

# --------- Functions ---------
def extract_model(model_path: str, extract_to: str = "."):
    tarball_path = os.path.join(model_path, "model.tar.gz")
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    logger.info(f"Model extracted to: {extract_to}")

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    try:
        vectorizer = CountVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([text1, text2])
        if vectors.shape[1] == 0:
            logger.warning("Empty vocabulary after stopword removal.")
            return 0.0
        return cosine_similarity(vectors[0], vectors[1])[0][0]
    except ValueError as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def check_span_overlap(model_span: dict, manual_span: dict) -> bool:
    model_start, model_end = model_span["start_char"], model_span["end_char"]
    manual_start, manual_end = manual_span["start_char"], manual_span["end_char"]

    if model_end >= manual_start and manual_end >= model_start:
        translator = str.maketrans("", "", string.punctuation)
        model_text = model_span.get("span_text", "").translate(translator).lower()
        manual_text = manual_span.get("span_text", "").translate(translator).lower()

        if model_text and manual_text:
            model_words = set(model_text.split())
            manual_words = set(manual_text.split())
            if model_words & manual_words:
                return calculate_cosine_similarity(model_text, manual_text) > THRESHOLD
    return False

def load_model_and_predict(extract_path: str, manually_coded_data: pd.DataFrame) -> pd.DataFrame:
    model_dir = os.path.join(extract_path, "output", "model-last")
    nlp = spacy.load(model_dir)
    logger.info(f"Loaded model from {model_dir}")

    rows = []
    manually_coded_df = manually_coded_data.drop_duplicates(subset=["unique_id"]).reset_index(drop=True)

    for _, row in manually_coded_df.iterrows():
        text = row["text"]
        unique_id = row["unique_id"]
        doc = nlp(text)
        spans = doc.spans.get("sc", [])

        for span, confidence in zip(spans, spans.attrs.get("scores", [])):
            rows.append({
                "text": text,
                "unique_id": unique_id,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "span": span.label_,
                "span_text": span.text,
                "score": confidence
            })

    model_df = pd.DataFrame(rows)
    return model_df

def remove_duplicated_spans(model_df: pd.DataFrame) -> pd.DataFrame:
    for unique_id in model_df["unique_id"].unique():
        comment_df = model_df[model_df["unique_id"] == unique_id]
        for i in range(len(comment_df)):
            for j in range(i + 1, len(comment_df)):
                if check_span_overlap(comment_df.iloc[i], comment_df.iloc[j]):
                    if len(comment_df.iloc[i]["span_text"]) > len(comment_df.iloc[j]["span_text"]):
                        model_df.at[comment_df.iloc[j].name, "score"] = 0
                    else:
                        model_df.at[comment_df.iloc[i].name, "score"] = 0

    return model_df[model_df["score"] > 0].reset_index(drop=True)

def evaluate_model(model_df: pd.DataFrame, manually_coded_data: pd.DataFrame) -> dict:
    model_df["true_positive"] = 0

    for unique_id in set(model_df["unique_id"]).union(manually_coded_data["unique_id"]):
        for label in set(model_df["span"]).union(manually_coded_data["span"]):
            model_spans = model_df[(model_df["unique_id"] == unique_id) & (model_df["span"] == label)]
            manual_spans = manually_coded_data[(manually_coded_data["unique_id"] == unique_id) & (manually_coded_data["span"] == label)]

            matched_model_indices = set()
            matched_manual_indices = set()

            for model_idx, model_span in model_spans.iterrows():
                for manual_idx, manual_span in manual_spans.iterrows():
                    if model_idx not in matched_model_indices and manual_idx not in matched_manual_indices:
                        if check_span_overlap(model_span, manual_span):
                            matched_model_indices.add(model_idx)
                            matched_manual_indices.add(manual_idx)
                            model_df.at[model_idx, "true_positive"] = 1
                            break

    model_df.drop_duplicates(inplace=True)

    true_positive = model_df["true_positive"].sum()
    false_positive = len(model_df) - true_positive
    false_negative = len(manually_coded_data) - true_positive
    true_negative = 0

    precision = round(true_positive / (true_positive + false_positive), 2) if (true_positive + false_positive) > 0 else 0.0
    recall = round(true_positive / (true_positive + false_negative), 2) if (true_positive + false_negative) > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 2) if (precision + recall) > 0 else 0.0

    y_scores = model_df["score"]
    y_true = model_df["true_positive"]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    roc_auc = round(auc(fpr, tpr), 2)

    return {
        "binary_classification_metrics": {
            "confusion_matrix": {
                "0": {"0": int(true_negative), "1": int(false_positive)},
                "1": {"0": int(false_negative), "1": int(true_positive)},
            },
            "precision": {"value": float(precision)},
            "recall": {"value": float(recall)},
            "f1": {"value": float(f1)},
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": [float(x) for x in fpr],
                "true_positive_rates": [float(x) for x in tpr],
            },
            "precision_recall_curve": {
                "precisions": [float(x) for x in precision_curve],
                "recalls": [float(x) for x in recall_curve],
            },
            "auc": {"value": float(roc_auc)},
        }
    }

def save_evaluation_results_to_s3(model_data: pd.DataFrame, bucket: str, key_prefix: str) -> str:
    s3_client = boto3.client("s3")
    csv_output_path = "/tmp/model_outputs.csv"
    model_data.to_csv(csv_output_path, index=False)

    s3_output_path = f"{key_prefix}/model_outputs.csv"
    try:
        s3_client.upload_file(csv_output_path, bucket, s3_output_path)
    except Exception as e:
        logger.error(f"Failed to upload file to S3: {e}")
        raise

    return f"s3://{bucket}/{s3_output_path}"

def create_s3_directory(bucket: str, key_prefix: str):
    boto3.client("s3").put_object(Bucket=bucket, Key=f"{key_prefix}/")

# --------- Main Execution ---------
if __name__ == "__main__":
    model_path = "/opt/ml/processing/model"
    test_path = "/opt/ml/processing/test/test.csv"
    extract_path = "model"

    extract_model(model_path, extract_to=extract_path)

    test_data = pd.read_csv(test_path)
    model_data = load_model_and_predict(extract_path, test_data)
    model_data = remove_duplicated_spans(model_data)

    evaluation_report = evaluate_model(model_data, test_data)
    logger.info(f"Evaluation report: {evaluation_report}")

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_output_path = os.path.join(output_dir, "model_outputs.csv")
    model_data.to_csv(csv_output_path, index=False)
    logger.info(f"CSV output saved to: {csv_output_path}")

    bucket = os.environ.get("BUCKET_NAME")
    key_prefix = os.environ.get("KEY_PREFIX")

    if not bucket or not key_prefix:
        raise ValueError("Both BUCKET_NAME and KEY_PREFIX environment variables must be set")

    create_s3_directory(bucket, key_prefix)
    csv_s3_uri = save_evaluation_results_to_s3(model_data, bucket, key_prefix)

    evaluation_report["csv_s3_uri"] = csv_s3_uri

    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(evaluation_report, f)

    logger.info(f"Evaluation complete. Results saved to: {evaluation_path}")
    logger.info(f"CSV output URI: {csv_s3_uri}")