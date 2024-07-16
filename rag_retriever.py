import csv
from typing import List, Dict, Tuple
from dspy.retrieve.qdrant_rm import QdrantRM
from qdrant_client import QdrantClient
import numpy as np

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

def main():
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

    vectorizers = [
        None,
        "sentence-transformers/all-MiniLM-L6-v2",
        "nomic-ai/nomic-embed-text-v1.5-Q",
        "BAAI/bge-large-en-v1.5",
        "intfloat/multilingual-e5-large"
    ]

    dataset_metrics = []
    all_results = []

    for vectorizer in vectorizers:
        model_name = vectorizer if vectorizer else "default"
        print(f"Evaluating with model: {model_name}")

        qdrant_retriever_model = build_retriever_client(labels=classes, collection_name="rad", k=3, vectorizer=vectorizer)
        results, mrr, recall = evaluate_retrieval(reports, ground_truth, qdrant_retriever_model)

        dataset_metrics.append({
            "model": model_name,
            "mean_reciprocal_rank": mrr,
            "recall_at_k": recall
        })

        for result in results:
            result["model"] = model_name
            all_results.append(result)

    # Save dataset-level metrics
    with open('dataset_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['model', 'mean_reciprocal_rank', 'recall_at_k']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset_metrics:
            writer.writerow(row)

    # Save sample-level results
    with open('sample_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['model', 'report', 'label', 'position']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print("Results have been saved to 'dataset_metrics.csv' and 'sample_results.csv'")

if __name__ == "__main__":
    main()