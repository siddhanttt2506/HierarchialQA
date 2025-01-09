import os
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import nltk

# NLTK downloads
nltk.download("wordnet")
nltk.download("punkt")

# Preload FAISS index and BM25 (placeholders)
FAISS_INDEX = None
BM25_INDEX = None
BM25_DOCS = []
METADATA = []

# -----------------------------------
# Function to Traverse Vectorized Tree
# -----------------------------------
def traverse_vectorized_tree(node, embeddings, metadata):
    """
    Traverse the vectorized tree and extract embeddings and metadata.
    """
    # Extract embeddings and metadata for the current node
    content_embedding = node.get("content_embedding")
    title_embedding = node.get("title_embedding")
    id_embedding = node.get("id_embedding")

    # Combine embeddings (use content embedding primarily, fall back to others if unavailable)
    combined_embedding = (
        np.mean(
            [
                emb
                for emb in [content_embedding, title_embedding, id_embedding]
                if emb is not None
            ],
            axis=0,
        )
        if any([content_embedding, title_embedding, id_embedding])
        else None
    )

    if combined_embedding is not None:
        embeddings.append(combined_embedding)
        metadata.append(
            {
                "id": node.get("id"),
                "title": node.get("title"),
                "content": node.get("content"),
            }
        )

    # Recursively process children
    for child in node.get("children", []):
        traverse_vectorized_tree(child, embeddings, metadata)

# -----------------------------------
# Initialization for FAISS and BM25
# -----------------------------------
def initialize_bm25(docs):
    """
    Initialize BM25 with tokenized documents.
    """
    global BM25_INDEX, BM25_DOCS
    BM25_DOCS = [doc["content"].split() for doc in docs if doc["content"]]
    BM25_INDEX = BM25Okapi(BM25_DOCS)


def initialize_faiss(embeddings):
    """
    Initialize FAISS with embeddings.
    """
    global FAISS_INDEX
    dim = embeddings.shape[1]
    FAISS_INDEX = faiss.IndexFlatL2(dim)
    FAISS_INDEX.add(embeddings)

# -----------------------------------
# Main Execution
# -----------------------------------
if __name__ == "__main__":
    # Path to your vectorized hierarchical tree folder
    folder_path = "vectorized_trees"
    embeddings = []  # This will store all dense embeddings
    metadata = []  # Metadata for each embedding

    # Process each vectorized tree
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):  # Process only JSON files
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                vectorized_tree = json.load(file)
                traverse_vectorized_tree(vectorized_tree, embeddings, metadata)

    # Convert embeddings list to numpy array
    embeddings = np.array(embeddings).astype("float32")

    # Initialize FAISS and BM25
    initialize_faiss(embeddings)
    initialize_bm25(metadata)

    print(f"FAISS index initialized with {len(embeddings)} embeddings.")
    print(f"BM25 index initialized with {len(metadata)} documents.")
