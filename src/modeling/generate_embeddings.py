import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import torch


# --- Configuration ---
CONFIG = {
    'PROCESSED_DATA_DIR': 'processed_data/',
    'OUTPUT_EMBEDDING_FILE': 'text_embeddings.npz',
    # 'all-MiniLM-L6-v2' is a great balance of speed and performance.
    'ST_MODEL_NAME': 'all-MiniLM-L6-v2'
}

# Ensure the output directory exists
os.makedirs(CONFIG['PROCESSED_DATA_DIR'], exist_ok=True)

if __name__ == '__main__':
    # --- 1. Load Processed Data ---
    print("Loading processed feature data to get the text column...")
    try:
        df_train = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features.csv'))
        df_test = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features_test.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run the main feature_engineering.py script first.")
        exit()

    output_path = os.path.join(CONFIG['PROCESSED_DATA_DIR'], CONFIG['OUTPUT_EMBEDDING_FILE'])

    # --- 2. Load the Pre-trained Transformer Model ---
    # This will download the model from the internet on the first run.
    print(f"Loading Sentence Transformer model: '{CONFIG['ST_MODEL_NAME']}'...")
    transformer_model = SentenceTransformer(CONFIG['ST_MODEL_NAME'])

    # --- 3. Prepare Text Data ---
    # Convert the text column to a list of strings, handling any potential missing values.
    print("Preparing text data for embedding...")
    train_texts = df_train['text_for_embedding'].fillna('missing').tolist()
    test_texts = df_test['text_for_embedding'].fillna('missing').tolist()

    # --- 4. Generate Embeddings (The Slow, Heavy-Lifting Step) ---
    print("Generating embeddings for the training data...")
    train_embeddings = transformer_model.encode(train_texts, show_progress_bar=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    print("Generating embeddings for the test data...")
    test_embeddings = transformer_model.encode(test_texts, show_progress_bar=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    # --- 5. Save the Embeddings ---
    print(f"\nSaving embeddings to compressed file: {output_path}")
    np.savez_compressed(output_path, train=train_embeddings, test=test_embeddings)

    print("\nEmbedding generation complete.")
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")