import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

# TODO models to test: sentence-transformers/all-MiniLM-L6-v2, sentence-transformers/all-MiniLM-L12-v2, sentence-transformers/all-mpnet-base-v2, intfloat/e5-small-v2, Alibaba-NLP/gte-Qwen2-7B-instruct, nomic-ai/nomic-embed-text-v1.5 
def get_embeddings(model, texts: List[str]):
    """
    Generates embeddings for a list of texts using the given model.
    """
    return model.encode(texts)

def compute_similarity(query_embedding, corpus_embeddings):
    return cosine_similarity([query_embedding], corpus_embeddings)

def main(corpus, query, model_name):
    
    corpus_embeddings = get_embeddings(model_name, corpus)
    query_embedding = get_embeddings(model_name, [query])[0]
    
    similarities = compute_similarity(query_embedding, corpus_embeddings)
    
    most_similar_index = np.argmax(similarities)
    print(f"Most similar sentence in corpus: {corpus[most_similar_index]}")
    print(f"Similarity score: {similarities[0][most_similar_index]}")

if __name__ == "__main__":
    # Example usage: python your_script.py --corpus "This is a sentence." "Here is another sentence." "Sentence embeddings are useful." --query "How are sentence embeddings calculated?" --model_name "sentence-transformers/all-MiniLM-L6-v2"

    parser = argparse.ArgumentParser(description="Embed and compute similarity between sentences.")
    
    parser.add_argument("--corpus", nargs='+', required=True, help="List of sentences in the corpus.")
    parser.add_argument("--query", type=str, required=True, help="Query sentence to compare against the corpus.")
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Name of the pre-trained model to use.")
    
    args = parser.parse_args()
    
    corpus = args.corpus
    query = args.query
    model_name = args.model_name

    main(corpus=corpus, query=query, model_name=model_name)
