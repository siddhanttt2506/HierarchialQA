import faiss
import numpy as np
import json
import os

# Paths
FOLDER_PATH = "vectorized_trees"  # Update this path to your folder with JSON files
FAISS_INDEX_PATH = "faiss_index_with_book_ids.bin"  # Path to save the FAISS index
METADATA_PATH = "metadata_with_book_ids.json"  # Path to save the metadata

def load_vectorized_trees(folder_path):
    """
    Load all vectorized hierarchical trees from a folder.
    """
    trees = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                tree = json.load(f)
                trees.append((os.path.splitext(file_name)[0], tree))  # Use file name (without extension) as book_id
    return trees

import numpy as np

def extract_embeddings_and_metadata(tree, book_id):
    """
    Extract embeddings and metadata from the hierarchical tree recursively.
    """
    embeddings = []
    metadata = []

    def traverse(node):
        # Extract embeddings
        content_embedding = np.array(node.get("content_embedding", []), dtype="float32")
        title_embedding = np.array(node.get("title_embedding", []), dtype="float32")
        id_embedding = np.array(node.get("id_embedding", []), dtype="float32")

        # Log embedding shapes for debugging
        print(f"Processing node {node.get('id')}")
        print(f"Content embedding shape: {content_embedding.shape}")
        print(f"Title embedding shape: {title_embedding.shape}")
        print(f"ID embedding shape: {id_embedding.shape}")

        # Filter out invalid embeddings (non-matching shapes or empty)
        valid_embeddings = [
            embedding for embedding in [content_embedding, title_embedding, id_embedding]
            if embedding.ndim == 1 and embedding.size > 0
        ]

        if valid_embeddings:
            try:
                # Combine embeddings using mean
                combined_embedding = np.mean(valid_embeddings, axis=0)
                embeddings.append(combined_embedding)
                metadata.append({
                    "id": node.get("id"),
                    "title": node.get("title"),
                    "content": node.get("content"),
                    "book_id": book_id,  # Add book identifier
                    "path": f"{book_id}/{node.get('id')}"  # Unique path for hierarchical traversal
                })
            except Exception as e:
                print(f"Error combining embeddings for node {node.get('id')}: {e}")
                return

        # Recursively process children
        for child in node.get("children", []):
            traverse(child)

    traverse(tree)
    return np.array(embeddings, dtype="float32"), metadata


def create_faiss_index(embeddings, dimension):
    """
    Create a FAISS index from the embeddings.
    """
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
    index.add(embeddings)
    return index

def save_faiss_index(index, metadata, index_path, metadata_path):
    """
    Save the FAISS index and metadata to disk.
    """
    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    # Load all vectorized hierarchical trees from the folder
    trees = load_vectorized_trees("vectorized_trees")
    all_embeddings = []
    all_metadata = []

    # Process each tree
    for book_id, tree in trees:
        print(f"Processing book: {book_id}...")
        embeddings, metadata = extract_embeddings_and_metadata(tree, book_id)
        all_embeddings.append(embeddings)
        all_metadata.extend(metadata)

    # Combine embeddings from all books
    if all_embeddings:
        combined_embeddings = np.vstack(all_embeddings)  # Stack all embeddings
        dimension = combined_embeddings.shape[1]

        # Create FAISS index
        faiss_index = create_faiss_index(combined_embeddings, dimension)

        # Save the FAISS index and metadata
        save_faiss_index(faiss_index, all_metadata, FAISS_INDEX_PATH, METADATA_PATH)
    else:
        print("No embeddings found to index.")
