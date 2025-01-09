import os
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import nltk
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import spacy
from gensim.models.phrases import Phrases, Phraser
import re
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

FAISS_INDEX_PATH = "faiss_index_with_book_ids.bin"
METADATA_PATH = "metadata_with_book_ids.json"
DENSE_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index from {index_path}")
    return index

# Load metadata
def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded metadata from {metadata_path}")
    return metadata

FAISS_INDEX = load_faiss_index(FAISS_INDEX_PATH)
METADATA = load_metadata(METADATA_PATH)


def generate(model, prompt):
    try:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
        return model.generate_content(prompt, safety_settings=safety_settings)
    except Exception as e:
        time.sleep(65)
        return generate(model, prompt)


def query_expansion(query, model):
    prompt = prompt = f"""
    You are an advanced query assistant helping to expand user queries for better information retrieval. 

    A query expansion improves the original query by adding:
    1. Relevant synonyms.
    2. Related concepts or terms.
    3. Important phrases that may help capture the intent of the query.
    4. Contextual phrases from related topics.

    The goal is to provide an expanded version of the query that captures the user's intent comprehensively. 
    For example:
    Original Query: "Explain semantic equivalence in propositional logic."
    Expanded Query: "semantic equivalence, propositional logic, logical equivalence, truth tables, tautology, logical connectives, predicate logic, soundness, completeness."

    Now, expand the following query comprehensively:
    Query: "{query}"

    Return the expanded query as a list of terms or phrases, separated by commas.
    """
    response = generate(model,prompt)

    expanded_query = response.candidates[0].content.parts[0].text.strip()
    terms = [term.strip() for term in expanded_query.split(",")]
    return list(terms)


def dense_retrieval(query,model, top_k=10):
    """
    Perform dense retrieval using FAISS with model-based query expansion.
    """
    expanded_query = " ".join(query_expansion(query,model))  # Expanded query
    query_embedding = DENSE_MODEL.encode(expanded_query).astype("float32").reshape(1, -1)
    distances, indices = FAISS_INDEX.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(METADATA):  # Ensure valid index
            results.append({
                "id": idx,
                "score": 1 - distances[0][i],  # Convert L2 distance to similarity
                "metadata": METADATA[idx]
            })
    return results


BM25_INDEX = None
BM25_DOCS = [doc["content"].split() for doc in METADATA if doc["content"]]

# Initialize BM25
BM25_INDEX = BM25Okapi(BM25_DOCS)

def bm25_retrieval(query,model, top_k=10):
    """
    Perform sparse retrieval using BM25.
    """
    expanded_query = query_expansion(query, model)  # Apply query expansion
    scores = BM25_INDEX.get_scores(expanded_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [{"id": idx, "score": scores[idx], "metadata": METADATA[idx]} for idx in top_indices]
    return results

def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    """
    Combine BM25 and dense retrieval results using RRF.
    """
    combined_scores = {}

    # Process BM25 results
    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Process Dense results
    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Sort by combined scores
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": doc_id, "score": score, "metadata": METADATA[doc_id]} for doc_id, score in sorted_results]

def generate_response(retrieved_docs, query):
    """
    Generate a response using an LLM.
    """
    context = "\n".join([f"{doc['metadata']['title']}: {doc['metadata']['content']}" for doc in retrieved_docs])
    prompt = f"Answer the following query based on the provided context:\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
    
    # Use a pre-configured LLM
    response = generate(model, prompt)  # Replace with your LLM's API call
    return response

def retrieval_pipeline(query,model, top_k=5):
    """
    Perform the full retrieval pipeline: query expansion, BM25, dense retrieval, RRF, and response generation.
    """
    # print("\n--- Query Expansion ---")
    # expanded_query = query_expansion(query,model)
    # print("Expanded Query:", expanded_query)

    print("\n--- Sparse Retrieval (BM25) ---")
    bm25_results = bm25_retrieval(query,model, top_k=top_k)
    print(f"BM25 Results: {len(bm25_results)} documents retrieved.")

    print("\n--- Dense Retrieval (FAISS) ---")
    dense_results = dense_retrieval(query,model, top_k=top_k)
    print(f"FAISS Results: {len(dense_results)} documents retrieved.")

    print("\n--- Reciprocal Rank Fusion (RRF) ---")
    combined_results = reciprocal_rank_fusion(bm25_results, dense_results)
    print(f"Final Combined Results: {len(combined_results)} documents retrieved.")

    return combined_results


if __name__ == "__main__":
    query = "Explain the role of ELIZA in NLP."
    retrieved_docs = retrieval_pipeline(query,model, top_k=5)

    print("\n--- Response Generation ---")
    response = generate_response(retrieved_docs, query)
    print("\nGenerated Response:")
    print(response)








