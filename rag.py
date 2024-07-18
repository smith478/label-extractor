import pandas as pd
from dspy.retrieve.qdrant_rm import QdrantRM
from qdrant_client import QdrantClient
import dspy
from typing import List, Dict, Tuple
import numpy as np
import csv
import os
import json

classes = ["pulmonary edema", "consolidation", "pleural effusion", "pneumothorax", "cardiomegaly"]
reports = [
    """RADIOLOGY REPORT
    Exam PA and lateral chest radiograph (2 views) (2 images) Date: XXXX, XXXX at XXXX hours Indication: Chest pain. Comparison: Chest radiograph from XXXX, XXXX. Findings: The cardiac silhouette is borderline enlarged. Otherwise, there is no focal opacity. Mediastinal contours are within normal limits. There is no large pleural effusion. No pneumothorax. Transcribed by - PSCB Transcription Date - XXXX
    IMPRESSION Borderline enlargement of the cardiac silhouette without acute pulmonary disease. DICTATED BY : Dr. XXXX XXXX XXXX XXXX XXXX ELECTRONICALLY SIGNED XXXX. XXXX XXXX XXXX XXXX XXXX TRANSCRIBED XXXX 11 XXXX XXXX RADRES XXXX
    SIGNATURE XXXX """,
    """RADIOLOGY REPORT
    Exam PA and lateral chest radiograph (2 views) (2 images) Date: XXXX, XXXX at XXXX hours Indication: Shortness of breath. Comparison: Chest radiograph from XXXX, XXXX. Findings: There is evidence of bilateral pulmonary edema. The cardiac silhouette is normal. No pleural effusion or pneumothorax. Transcribed by - PSCB Transcription Date - XXXX
    IMPRESSION Bilateral pulmonary edema. No evidence of pleural effusion or pneumothorax. DICTATED BY : Dr. XXXX XXXX XXXX XXXX XXXX ELECTRONICALLY SIGNED XXXX. XXXX XXXX XXXX XXXX XXXX TRANSCRIBED XXXX 11 XXXX XXXX RADRES XXXX
    SIGNATURE XXXX """
]
ground_truth = [["cardiomegaly"], ["pulmonary edema"]]

class ClassifyText(dspy.Signature):
    """Classify the radiology into multiple labels from the given candidates. You should return the 
    extracted information as a single JSON string with a key for each candidate label and a value of
    1 if the report indicates the presence of the abnormality and 0 otherwise. There should be no 
    text or explanation, only the JSON. For example if there 
    were 3 candidates you could have the following output:

    {
        "label_1": 1,
        "label_2": 0,
        "label_3": 1
    }"""
    text = dspy.InputField()
    label_candidates = dspy.InputField(desc="List of candidate labels for the text")
    rad_labels = dspy.OutputField(desc="Dictionary of candidate labels, 1 or 0, for the text")

class RAGMultiLabelClassifier(dspy.Module):
    def __init__(self, num_candidates=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_candidates)
        self.classify = dspy.Predict(ClassifyText)

    def forward(self, text):
        retrieved_docs = ','.join(self.retrieve(text).passages)
        classification_result = self.classify(text=text, label_candidates=retrieved_docs)
        return classification_result.rad_labels
    
def build_retriever_client(labels: List[str], collection_name: str, k: int, vectorizer: str = None) -> QdrantRM:
    client = QdrantClient(":memory:")
    ids = list(range(len(labels)))
    
    if vectorizer:
        client.set_model(vectorizer)
        
    client.add(
        collection_name=collection_name,
        documents=labels,
        ids=ids
    )
    return QdrantRM(collection_name, client, k=k)

def evaluate_retrieval(reports: List[str], ground_truth: List[List[str]], retriever_model: QdrantRM, k: int = 5) -> Tuple[List[Dict], float, float]:
    results = []
    positions = []
    top_k = 0

    for report, labels in zip(reports, ground_truth):
        retrieval_results = retriever_model.forward(report, k=k)
        results_list = [elt['long_text'] for elt in retrieval_results]

        for label in labels:
            if label in results_list:
                position = results_list.index(label) + 1
                top_k += 1
            else:
                position = k + 1  # Setting to k+1 if not found within top k
            
            positions.append(position)
            results.append({
                "report": report[:50],  # Truncating report for brevity
                "label": label,
                "position": position
            })

    mean_reciprocal_rank = np.mean([1/p for p in positions])
    recall_at_k = top_k / len(positions)

    return results, mean_reciprocal_rank, recall_at_k

def clean_json_string(json_str: str) -> str:
    # Remove the backticks and the "json" text
    return json_str.replace('```json\n', '').replace('\n```', '')

def parse_ollama_output(output_str: str, clean_values: bool = True) -> List[str]:
    if clean_values:
        # Remove the backticks and the "json" text
        output_str = clean_json_string(output_str)
    output_dict = json.loads(output_str)
    predicted_classes = [key for key, value in output_dict.items() if value == 1]
    return predicted_classes

def calculate_metrics(ground_truth: List[List[str]], predicted_classes: List[str]) -> Dict[str, float]:
    tp, fp, fn = 0, 0, 0

    for gt_labels, pred_labels in zip(ground_truth, predicted_classes):
        gt_set = set(gt_labels)
        pred_set = set(pred_labels)

        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

def save_metrics(dataset_metrics: List[Dict[str, float]], file_path: str):
    keys = dataset_metrics[0].keys()
    with open(file_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset_metrics)

def main():
    vectorizers = [
        None,
        "sentence-transformers/all-MiniLM-L6-v2",
        "nomic-ai/nomic-embed-text-v1.5-Q",
        "BAAI/bge-large-en-v1.5",
        "intfloat/multilingual-e5-large"
    ]
    ollama_models = [
        {"model": "llama3", "type": "text"},
        {"model": "internlm2", "type": "text"},
        {"model": "gemma2", "type": "text"},
    ]

    dataset_metrics = []

    for vectorizer in vectorizers:
        for ollama in ollama_models:
            vectorizer_name = vectorizer if vectorizer else "default"
            print(f"Calculating performance for RM: {vectorizer_name} and LM: {ollama['model']}")
            retriever_model = build_retriever_client(labels=classes, collection_name="rad", k=3, vectorizer=vectorizer)
            ollama_model = dspy.OllamaLocal(
                model=ollama['model'], 
                model_type=ollama['type'],
                max_tokens=512,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                top_k=3,
                format='json'
            )

            dspy.settings.configure(lm=ollama_model, rm=retriever_model)
            classifier = RAGMultiLabelClassifier(num_candidates=3)

            predictions = []

            for report, labels in zip(reports, ground_truth):
                result_str = classifier(text=report)
                try:
                    predicted_classes = parse_ollama_output(result_str)
                    predictions.append(predicted_classes)
                except json.JSONDecodeError:
                    print("Warning! Could not parse output from Ollama. Skipping this result.")
                    print(f'Report: {report}')
                    print(f'Result string: {result_str}')
                    continue

            metrics = calculate_metrics(ground_truth, predictions)

            dataset_metrics.append({
                "vectorizer": vectorizer_name,
                "ollama_model": ollama['model'],
                **metrics
            })

    save_metrics(dataset_metrics, 'dataset_metrics.csv')
    print("Results have been saved to dataset_metrics.csv")

if __name__ == '__main__':
    main()
