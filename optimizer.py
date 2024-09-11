import pandas as pd
from dspy.retrieve.qdrant_rm import QdrantRM
from qdrant_client import QdrantClient
import dspy
from typing import List, Dict, Tuple
import numpy as np
import csv
import os
import json

def clean_json_string(json_str: str) -> str:
    # Remove the backticks and the "json" text
    return json_str.replace('```json\n', '').replace('\n```', '')

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
        result = classification_result.rad_labels
        result = clean_json_string(result)
        # Parse the JSON string into a dictionary
        return json.loads(result)
    
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

def parse_ollama_output(output_str: str, clean_values: bool = True) -> List[str]:
    if clean_values:
        # Remove the backticks and the "json" text
        output_str = clean_json_string(output_str)
    output_dict = json.loads(output_str)
    predicted_classes = [key for key, value in output_dict.items() if value == 1]
    return predicted_classes

retriever_model = build_retriever_client(labels=classes, 
                                         collection_name="rad", 
                                         k=3, 
                                         vectorizer=vectorizer)

class CustomOllamaLocal(dspy.OllamaLocal):
    def __init__(self, model, **kwargs):
        logger.debug(f"Initializing CustomOllamaLocal with model: {model}")
        self.model = model  # Explicitly set the model attribute
        super().__init__(model=model, **kwargs)
        
    def copy(self, **kwargs):
        logger.debug(f"Copying CustomOllamaLocal with kwargs: {kwargs}")
        new_kwargs = self.__dict__.copy()
        new_kwargs.update(kwargs)
        return CustomOllamaLocal(**new_kwargs)
    
    def basic_request(self, prompt, **kwargs):
        logger.debug(f"Making basic request with model: {self.model}")
        return super().basic_request(prompt, **kwargs)
    
ollama_model = CustomOllamaLocal(
    model=ollama_model_name, 
    model_type='text',
    max_tokens=512,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    top_k=5,
    format='json'
)

dspy.settings.configure(lm=ollama_model, rm=retriever_model)
classifier = RAGMultiLabelClassifier(num_candidates=3)

def accuracy(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)
    return int(pred_set == gold_set)

def evaluate_model(model, dataset):
    total_accuracy = 0
    for example in dataset:
        result = model(text=example['text'])
        print(result)
        if isinstance(result, str):
            predicted_classes = parse_ollama_output(result)
        else:
            predicted_classes = [k for k, v in result.items() if v == 1]
        total_accuracy += accuracy(predicted_classes, example['ground_truth'])
    return total_accuracy / len(dataset)

data = [dspy.Example(ground_truth=label, text=report).with_inputs("text") for report, label in zip(reports, ground_truth)]

trainset = data[:20]
devset = data[20:30]
testset = data[30:]

def metric(gold, pred, trace=None):
    if isinstance(pred, str):
        predicted_classes = parse_ollama_output(pred)
    else:
        predicted_classes = [k for k, v in pred.items() if v == 1]
    acc = accuracy(predicted_classes, gold.ground_truth)
    return acc

def custom_merge_dicts(d1, d2):
    merged = d1.copy()
    if isinstance(d2, str):
        merged['prediction'] = d2
    elif isinstance(d2, dict):
        for k, v in d2.items():
            if k in d1:
                merged[f"pred_{k}"] = v
            else:
                merged[k] = v
    return merged

from dspy.teleprompt import BootstrapFewShotWithRandomSearch

teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, 
                                                max_bootstrapped_demos=4,
                                                max_labeled_demos=4,
                                                max_rounds=3,
                                                num_candidate_programs=2,
                                                num_threads=2
                                               )

import dspy.evaluate.evaluate as dspy_evaluate

# Save the original function
original_merge_dicts = dspy_evaluate.merge_dicts

# Replace with your custom function
dspy_evaluate.merge_dicts = custom_merge_dicts

# Now run your compilation
compiled_rag_2 = teleprompter.compile(classifier, trainset=trainset)

# After you're done, you can restore the original function if needed
# dspy_evaluate.merge_dicts = original_merge_dicts