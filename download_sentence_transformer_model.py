from sentence_transformers import SentenceTransformer
import os
import argparse

def download_sentence_transformer_model(model_name, save_directory):
    """
    Downloads a Sentence Transformer model and saves it to disk.

    Args:
    model_name (str): The name of the pre-trained Sentence Transformer model to download.
    save_directory (str): The directory where the model will be saved, with model_name appended.
    """
    # Append model_name to save_directory
    save_directory = os.path.join(save_directory, model_name)
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Download and save the model
    model = SentenceTransformer(model_name)
    model.save(save_directory)

    print(f"Sentence Transformer model saved to {save_directory}")

if __name__ == "__main__":
    # example usage: python download_sentence_transformer_model.py --model_name "sentence-transformers/all-MiniLM-L6-v2" --save_directory "./models"
    parser = argparse.ArgumentParser(description="Download a Sentence Transformer model and save it to disk.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the pre-trained Sentence Transformer model to download.")
    parser.add_argument("--save_directory", type=str, required=True, help="The base directory where the model will be saved.")
    
    args = parser.parse_args()
    
    download_sentence_transformer_model(args.model_name, args.save_directory)