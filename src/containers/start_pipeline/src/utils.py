import spacy
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THRESHOLD = 0.2

def calculate_cosine_similarity(text1, text2):
    try:
        vectorizer = CountVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([text1, text2])
        
        if vectors.shape[1] == 0:
            logger.warning("Empty vocabulary after removing stop words. Skipping similarity calculation.")
            return 0.0  # Skip and return similarity of 0.0
        
        cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        return cosine_sim
    
    except ValueError as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0  # Skip and return similarity of 0.0


def check_span_overlap(model_span, manual_span):
    model_start, model_end = model_span['start_char'], model_span['end_char']
    manual_start, manual_end = manual_span['start_char'], manual_span['end_char']

    if not (model_end < manual_start or manual_end < model_start):
        translator = str.maketrans('', '', string.punctuation)
        model_text = model_span.get('span_text', '').translate(translator).lower()
        manual_text = manual_span.get('span_text', '').translate(translator).lower()

        if not model_text or not manual_text:
            logger.warning(f"Empty span text detected. Model text: '{model_text}', Manual text: '{manual_text}'")
            return False

        model_words = set(model_text.split())
        manual_words = set(manual_text.split())
        if model_words & manual_words:
            cosine_sim = calculate_cosine_similarity(model_text, manual_text)
            if cosine_sim > THRESHOLD:
                return True
    return False


def load_model_and_predict(model_dir, manually_coded_data):
    nlp = spacy.load(model_dir)
    print("Loaded model from", model_dir)

    model_df = pd.DataFrame(columns=['text', 'unique_id', 'start_char', 'end_char', 'label', 'span_text', 'score'])
    manually_coded_df = manually_coded_data.drop_duplicates(subset=['unique_id']).reset_index(drop=True)
    for index, row in manually_coded_df.iterrows():
        text = row['text']
        unique_id = row['unique_id']
        doc = nlp(text)
        spans = doc.spans['sc']

        for span, confidence in zip(spans, spans.attrs["scores"]):
            start_char = span.start_char
            end_char = span.end_char
            span = span.label_
            span_text = span.text
            score = confidence
            model_df = pd.concat([model_df, pd.DataFrame([[text, unique_id, start_char, end_char, span, span_text, score]], columns=['text', 'unique_id', 'start_char', 'end_char', 'span', 'span_text', 'score'])], ignore_index=True)
    
    logger.info(f"Model data columns after prediction: {model_df.columns}")
    
    if 'span' not in model_df.columns:
        model_df['span'] = None  

    return model_df


def remove_duplicated_spans(model_df):
    for unique_id in model_df['unique_id'].unique():
        comment_df = model_df[model_df['unique_id'] == unique_id]
        for i in range(len(comment_df)):
            for j in range(i+1, len(comment_df)):
                if check_span_overlap(comment_df.iloc[i], comment_df.iloc[j]):
                    if len(comment_df.iloc[i]['span_text']) > len(comment_df.iloc[j]['span_text']):
                        # remove the shorter span from model_df based on span_text, unique_id
                        model_df.at[comment_df.iloc[j].name, 'score'] = 0
                    else:
                        model_df.at[comment_df.iloc[i].name, 'score'] = 0

    model_df = model_df[model_df['score'] > 0].reset_index(drop=True)
    return model_df



def evaluate_model(model_df, manually_coded_data):
    model_df['true_positive'] = 0

    for unique_id in set(model_df['unique_id']).union(set(manually_coded_data['unique_id'])):
        for label in set(model_df['span']).union(set(manually_coded_data['span'])):
            model_spans = model_df[(model_df['unique_id'] == unique_id) & (model_df['span'] == label)]
            manual_spans = manually_coded_data[(manually_coded_data['unique_id'] == unique_id) & (manually_coded_data['span'] == label)]

            matched_model_indices = set()
            matched_manual_indices = set()

            for model_idx, model_span in model_spans.iterrows():
                for manual_idx, manual_span in manual_spans.iterrows():
                    if manual_idx not in matched_manual_indices and model_idx not in matched_model_indices:
                        if check_span_overlap(model_span, manual_span):
                            matched_model_indices.add(model_idx)
                            matched_manual_indices.add(manual_idx)
                            model_df.at[model_idx, 'true_positive'] = 1
                            break

    model_df.drop_duplicates(inplace=True)

    true_positive = model_df['true_positive'].sum()
    false_positive = len(model_df) - true_positive
    false_negative = len(manually_coded_data) - true_positive
    true_negative = 0

    precision = round(true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0, 2)
    recall = round(true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0, 2)
    f1 = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0, 2)

    y_scores = model_df['score']
    y_true = model_df['true_positive'] 

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)

    roc_auc = round(auc(fpr, tpr), 2)

    # convert all metrics to float for json serialization
    roc_auc = float(roc_auc)  
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)
    true_positive = int(true_positive)
    false_positive = int(false_positive)
    false_negative = int(false_negative)
    true_negative = int(true_negative)

    fpr = [float(x) for x in fpr]
    tpr = [float(x) for x in tpr]
    precision_curve = [float(x) for x in precision_curve]
    recall_curve = [float(x) for x in recall_curve]


    return {
        "binary_classification_metrics": {  
            "confusion_matrix": {
                "0": {
                    "0": true_negative,
                    "1": false_positive
                },
                "1": {
                    "0": false_negative,
                    "1": true_positive
                }
            },
            "precision": {
                "value": precision,
            },
            "recall": {
                "value": recall,
            },
            "f1": {
                "value": f1,
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": fpr,
                "true_positive_rates": tpr
            },
            "precision_recall_curve": {
                "precisions": precision_curve,
                "recalls": recall_curve
            },
            "auc": {
                "value": roc_auc,
            },
        }   
    }