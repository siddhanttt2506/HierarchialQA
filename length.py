import json

with open("vectorized_hierarchical_tree.json", "r", encoding="utf-8") as f:
    vectorized_tree = json.load(f)

# Check the shape of title_embedding or content_embedding for any node
example_embedding = vectorized_tree["title_embedding"]
print(len(example_embedding))  # Should output 384 for `all-MiniLM-L6-v2`

