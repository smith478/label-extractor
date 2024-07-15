import numpy as np
import csv
from typing import List, Dict, Tuple
from dsp.modules import sentence_vectorizer
from dspy.retrieve.qdrant_rm import QdrantRM
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

def create_qdrant_collection(client: QdrantClient, collection_name: str, docs: List[str], vectorizer) -> None:
    embedded_docs = [vectorizer(doc) for doc in docs]
    vector_size = len(embedded_docs[0])

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance="Cosine")
    )

    points = [
        PointStruct(
            id=idx,
            vector=embedded_doc.tolist(),
            payload={"text": doc}
        )
        for idx, (doc, embedded_doc) in enumerate(zip(docs, embedded_docs))
    ]

    client.upsert(
        collection_name=collection_name,
        points=points
    )

def evaluate_retrieval(reports: List[str], ground_truth: List[List[str]], retriever_model: QdrantRM, k: int = 5) -> List[Tuple[str, str, int]]:
    results = []

    for report, labels in zip(reports, ground_truth):
        retrieval_results = retriever_model.forward(report, k=k)
        results_list = [elt['long_text'] for elt in retrieval_results]

        for label in labels:
            if label in results_list:
                position = results_list.index(label) + 1
            else:
                position = k + 1  # Setting to k+1 if not found within top k
            results.append((report[:50], label, position))  # Truncating report for brevity

    return results

def main():
    classes = ["pulmonary edema", "consolidation", "pleural effusion", "pneumothorax", "cardiomegaly"]

    reports = [
        """
        RADIOLOGY REPORT

        Exam
        PA and lateral chest radiograph (2 views) (2 images) Date: XXXX, XXXX at XXXX hours Indication: Chest pain. Comparison: Chest radiograph from XXXX, XXXX. Findings: The cardiac silhouette is borderline enlarged. Otherwise, there is no focal opacity. Mediastinal contours are within normal limits. There is no large pleural effusion. No pneumothorax. Transcribed by - PSCB Transcription Date - XXXX

        IMPRESSION
        Borderline enlargement of the cardiac silhouette without acute pulmonary disease. DICTATED BY : Dr. XXXX XXXX XXXX XXXX XXXX ELECTRONICALLY SIGNED XXXX. XXXX XXXX XXXX XXXX XXXX TRANSCRIBED XXXX 11 XXXX XXXX  RADRES XXXX

        SIGNATURE
        XXXX
        """,
        """
        RADIOLOGY REPORT

        Exam
        PA and lateral chest radiograph (2 views) (2 images) Date: XXXX, XXXX at XXXX hours Indication: Shortness of breath. Comparison: Chest radiograph from XXXX, XXXX. Findings: There is evidence of bilateral pulmonary edema. The cardiac silhouette is normal. No pleural effusion or pneumothorax. Transcribed by - PSCB Transcription Date - XXXX

        IMPRESSION
        Bilateral pulmonary edema. No evidence of pleural effusion or pneumothorax. DICTATED BY : Dr. XXXX XXXX XXXX XXXX XXXX ELECTRONICALLY SIGNED XXXX. XXXX XXXX XXXX XXXX XXXX TRANSCRIBED XXXX 11 XXXX XXXX  RADRES XXXX

        SIGNATURE
        XXXX
        """
    ]

    ground_truth = [
        ["cardiomegaly"],
        ["pulmonary edema"]
    ]

    embedding_models = [
        "all-mpnet-base-v2",
        "all-MiniLM-L6-v2",
        "paraphrase-multilingual-MiniLM-L12-v2"
    ]

    results = []

    for model_name in embedding_models:
        print(f"Evaluating with model: {model_name}")
        vectorizer = sentence_vectorizer.SentenceTransformersVectorizer(model_name)
        client = QdrantClient(":memory:")

        create_qdrant_collection(client, "rad", classes, vectorizer)
        qdrant_retriever_model = QdrantRM("rad", client, k=5)

        evaluation_results = evaluate_retrieval(reports, ground_truth, qdrant_retriever_model)
        
        for report, label, position in evaluation_results:
            results.append({
                "model": model_name,
                "report": report,
                "label": label,
                "position": position
            })

    # Save results to CSV
    with open('embedding_model_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['model', 'report', 'label', 'position']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Results have been saved to 'embedding_model_results.csv'")

if __name__ == "__main__":
    main()