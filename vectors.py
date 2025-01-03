


import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/gtr-t5-large')  # You can change the model if required.

def encode_node(node, model):
    """Recursively encode the content, title, ID, and summary of the hierarchical tree."""
    encoded_node = {
        "id": node.get("id"),
        "title": node.get("title"),
        "content": node.get("content"),
        # "summary": node.get("summary"),
        "id_embedding": None,
        "title_embedding": None,
        "content_embedding": None,
        # "summary_embedding": None,
        "children": []
    }
    
    # Encode the ID if available
    if node.get("id"):
        id_embedding = model.encode(node["id"], convert_to_tensor=True)
        encoded_node["id_embedding"] = id_embedding.cpu().numpy().tolist()
    
    # Encode the title if available
    if node.get("title"):
        title_embedding = model.encode(node["title"], convert_to_tensor=True)
        encoded_node["title_embedding"] = title_embedding.cpu().numpy().tolist()
    
    # Encode the content if available
    if node.get("content"):
        content_embedding = model.encode(node["content"], convert_to_tensor=True)
        encoded_node["content_embedding"] = content_embedding.cpu().numpy().tolist()
    
    # Encode the summary if available
    # if node.get("summary"):
    #     summary_embedding = model.encode(node["summary"], convert_to_tensor=True)
    #     encoded_node["summary_embedding"] = summary_embedding.cpu().numpy().tolist()
    
    # Recursively encode children
    for child in node.get("children", []):
        encoded_child = encode_node(child, model)
        encoded_node["children"].append(encoded_child)
    
    return encoded_node

# Define input and output file paths using os.path.join
input_file = os.path.join(BASE_DIR, "hierarchical_tree_with_summaries (1).json")
output_file = os.path.join(BASE_DIR, "vectorized_hierarchical_tree(1).json")

# Load the hierarchical tree JSON
with open(input_file, 'r', encoding='utf-8') as f:
    hierarchical_tree = json.load(f)

# Perform vectorization
vectorized_tree = encode_node(hierarchical_tree, model)

# Save the vectorized tree to a JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(vectorized_tree, f, ensure_ascii=False, indent=4)

print(f"Vectorized hierarchical tree saved to {output_file}")
