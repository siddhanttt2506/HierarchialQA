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

# Constants for file paths
FAISS_INDEX_PATH = "faiss_index_with_book_ids.bin"
METADATA_PATH = "metadata_with_book_ids.json"

# Load environment variables and configure Gemini
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize NLTK and models
nltk.download("wordnet")
nltk.download("punkt")
DENSE_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# FAISS and BM25 placeholders
FAISS_INDEX = None
BM25_INDEX = None
BM25_DOCS = []
METADATA = []

# RRF Parameter
RRF_K = 60

def dense_retrieval(query, top_k=10):
    global FAISS_INDEX, METADATA
    
    # Get query embedding
    query_embedding = DENSE_MODEL.encode([query]).astype('float32')
    
    # Get dimension of FAISS index
    d = FAISS_INDEX.d
    
    # Ensure query embedding matches index dimension
    if query_embedding.shape[1] != d:
        raise ValueError(f"Query embedding dimension ({query_embedding.shape[1]}) doesn't match index dimension ({d})")
    
    # Search
    distances, indices = FAISS_INDEX.search(query_embedding, top_k)
    
    return [{"id": idx, 
             "score": 1 - distances[0][i],
             "metadata": METADATA[idx]} 
            for i, idx in enumerate(indices[0]) 
            if idx < len(METADATA)]

# Load existing FAISS index and metadata
def load_existing_indices():
    global FAISS_INDEX, METADATA, BM25_INDEX
    
    # Load FAISS index
    FAISS_INDEX = faiss.read_index("faiss_index_with_book_ids.bin")
    
    # Load metadata with UTF-8 encoding
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        METADATA = json.load(f)
    
    # Initialize BM25
    BM25_DOCS = [doc["content"].split() for doc in METADATA if doc["content"]]
    BM25_INDEX = BM25Okapi(BM25_DOCS)
    
    return FAISS_INDEX, METADATA, BM25_INDEX

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


# -----------------------------------
# Query Expansion
# -----------------------------------
def query_expansion(query):
    tokens = word_tokenize(query)
    expanded_query = set(tokens)  # Include original tokens
    
    for token in tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                expanded_query.add(lemma.name())
    
    return list(expanded_query)


# -----------------------------------
# Sparse Retrieval (BM25)
# -----------------------------------
def bm25_retrieval(query, top_k=10):
    global BM25_INDEX, BM25_DOCS
    if BM25_INDEX is None:
        raise ValueError("BM25 index not initialized.")
    
    expanded_query = query_expansion(query)
    scores = BM25_INDEX.get_scores(expanded_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = [{"id": idx, "score": scores[idx], "metadata": METADATA[idx]} for idx in top_indices]
    return results


# -----------------------------------
# Dense Retrieval (FAISS)
# -----------------------------------
def dense_retrieval(query, top_k=10):
    global FAISS_INDEX, METADATA
    if FAISS_INDEX is None:
        raise ValueError("FAISS index not initialized.")
    
    query_embedding = DENSE_MODEL.encode(query).astype("float32").reshape(1, -1)
    distances, indices = FAISS_INDEX.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(METADATA):  # Ensure index is within bounds
            results.append({
                "id": idx,
                "score": 1 - distances[0][i],  # Convert distance to similarity
                "metadata": METADATA[idx]
            })
    return results


# -----------------------------------
# Reciprocal Rank Fusion (RRF)
# -----------------------------------
def reciprocal_rank_fusion(bm25_results, dense_results, k=RRF_K):
    combined_scores = {}
    
    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": doc_id, "score": score, "metadata": METADATA[doc_id]} for doc_id, score in sorted_results]


# -----------------------------------
# LLM-Based Response Generation
# -----------------------------------
def generate_response(retrieved_docs, query):
    """
    Generate a response using LLM, considering retrieved documents and the query.
    """
    context = "\n".join([f"{doc['metadata']['title']}: {doc['metadata']['content']}" for doc in retrieved_docs])
    prompt = f"Answer the following query based on the provided context:\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
    response = LLM(prompt, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]


# -----------------------------------
# Retrieval Pipeline
# -----------------------------------
def retrieval_pipeline(query, top_k=5):
    print("\n--- Query Expansion ---")
    expanded_query = query_expansion(query)
    print("Expanded Query:", expanded_query)

    print("\n--- Sparse Retrieval (BM25) ---")
    bm25_results = bm25_retrieval(query, top_k=top_k)
    print(f"BM25 Results: {len(bm25_results)} documents retrieved.")

    print("\n--- Dense Retrieval (FAISS) ---")
    dense_results = dense_retrieval(query, top_k=top_k)
    print(f"FAISS Results: {len(dense_results)} documents retrieved.")

    print("\n--- Reciprocal Rank Fusion (RRF) ---")
    combined_results = reciprocal_rank_fusion(bm25_results, dense_results)
    print(f"Final Combined Results: {len(combined_results)} documents retrieved.")

    return combined_results


# -----------------------------------
# Initialization
# -----------------------------------
def initialize_bm25(docs):
    global BM25_INDEX, BM25_DOCS
    BM25_DOCS = [doc["content"].split() for doc in docs if doc["content"]]
    BM25_INDEX = BM25Okapi(BM25_DOCS)


def initialize_faiss(embeddings):
    global FAISS_INDEX
    dim = embeddings.shape[1]
    FAISS_INDEX = faiss.IndexFlatL2(dim)
    FAISS_INDEX.add(embeddings)


# -----------------------------------
# Main
# -----------------------------------
# if __name__ == "__main__":
#     # Load hierarchical trees
#     folder_path = "vectorized_trees"
#     embeddings = []
#     metadata = []

#     def collect_documents(node):
#         if "content_embedding" in node:
#             embeddings.append(node["content_embedding"])
#             metadata.append({"id": node["id"], "title": node["title"], "content": node["content"]})
#         for child in node.get("children", []):
#             collect_documents(child)

#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".json"):
#             with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
#                 tree = json.load(file)
#                 collect_documents(tree)

#     # Convert embeddings to numpy array
#     embeddings = np.array(embeddings).astype("float32")

#     # Initialize retrieval systems
#     initialize_faiss(embeddings)
#     initialize_bm25(metadata)

#     print(f"FAISS initialized with {len(embeddings)} embeddings.")
#     print(f"BM25 initialized with {len(metadata)} documents.")

#     # Query Example
#     query = "Explain the role of ELIZA in NLP."
#     retrieved_docs = retrieval_pipeline(query, top_k=5)

#     print("\n--- Response Generation ---")
#     response = generate_response(retrieved_docs, query)
#     print("\nGenerated Response:")
#     print(response)
def generate_response(retrieved_docs, query):
    context = "\n".join([
        f"Book: {doc['metadata']['book_id']}\n"
        f"Section: {doc['metadata']['title']}\n"
        f"Content: {doc['metadata']['content']}"
        for doc in retrieved_docs[:3]
    ])
    
    prompt = f"""Based on the following context, answer this query: {query}

Context:
{context}

Please provide a clear and accurate answer based solely on the information provided above."""
    
    response = generate(model, prompt)
    return response.text

if __name__ == "__main__":
    # Load existing indices instead of creating new ones
    FAISS_INDEX, METADATA, BM25_INDEX = load_existing_indices()
    
    print(f"Loaded FAISS index with {FAISS_INDEX.ntotal} vectors")
    print(f"Loaded {len(METADATA)} metadata entries")
    
    # Example query
    query = "What is ELIZA?"
    retrieved_docs = retrieval_pipeline(query, top_k=5)
    
    print("\nGenerating response...")
    response = generate_response(retrieved_docs, query)
    print("\nResponse:", response)

#     nlp = spacy.load("en_core_web_sm")

# def query_expansion(query, metadata, top_k=10):
#     # Initialize expanded terms
#     expanded_terms = set()
    
#     # Process query
#     doc = nlp(query.lower())
#     query_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
#     # Extract key phrases from metadata
#     corpus_phrases = []
#     for item in metadata:
#         if item['content']:
#             corpus_phrases.extend(extract_phrases(item['content'].lower()))
    
#     # Train phrase detector
#     bigram = Phrases(corpus_phrases, min_count=2, threshold=5)
#     bigram_mod = Phraser(bigram)
    
#     # Add original query terms
#     expanded_terms.update(query_tokens)
    
#     # Add noun phrases from query
#     for chunk in doc.noun_chunks:
#         expanded_terms.add(chunk.text.lower())
    
#     # Find context-relevant terms
#     for term in query_tokens:
#         # Find related terms in corpus
#         related = find_related_terms(term, metadata, top_k)
#         expanded_terms.update(related)
        
#         # Add domain-specific collocations
#         collocations = find_collocations(term, bigram_mod, corpus_phrases)
#         expanded_terms.update(collocations)
    
#     # Clean and normalize terms
#     cleaned_terms = clean_terms(expanded_terms)
    
#     return list(cleaned_terms)

# def extract_phrases(text):
#     doc = nlp(text)
#     phrases = []
    
#     # Extract noun phrases
#     phrases.extend([chunk.text.lower() for chunk in doc.noun_chunks])
    
#     # Extract verb phrases
#     for token in doc:
#         if token.pos_ == "VERB":
#             phrase = token.text
#             for child in token.children:
#                 if child.dep_ in ['dobj', 'pobj']:
#                     phrase += ' ' + child.text
#             phrases.append(phrase.lower())
    
#     return phrases

# def find_related_terms(term, metadata, top_k):
#     related = set()
#     term_pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
    
#     # Find sentences containing the term
#     for item in metadata:
#         if item['content']:
#             sentences = [sent for sent in nlp(item['content']).sents 
#                         if term_pattern.search(sent.text.lower())]
            
#             # Extract key terms from these sentences
#             for sent in sentences:
#                 doc = nlp(sent.text)
#                 for token in doc:
#                     if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
#                         not token.is_stop and 
#                         token.text.lower() != term):
#                         related.add(token.text.lower())
    
#     # Sort by frequency and return top-k
#     return sorted(list(related))[:top_k]

# def find_collocations(term, bigram_mod, corpus_phrases):
#     collocations = set()
    
#     # Find bigrams containing the term
#     for phrase in corpus_phrases:
#         tokens = phrase.split()
#         bigrams = bigram_mod[tokens]
#         for bigram in bigrams:
#             if isinstance(bigram, tuple) and term in bigram:
#                 collocations.add(' '.join(bigram))
    
#     return collocations

# def clean_terms(terms):
#     cleaned = set()
#     for term in terms:
#         # Remove special characters and extra spaces
#         cleaned_term = re.sub(r'[^\w\s]', ' ', term)
#         cleaned_term = re.sub(r'\s+', ' ', cleaned_term).strip()
        
#         # Add if term is meaningful
#         if (cleaned_term and 
#             len(cleaned_term) > 2 and 
#             not cleaned_term.isnumeric()):
#             cleaned.add(cleaned_term)
    
#     return cleaned

# print(query_expansion("what is propositional logic and semantic equivalence", METADATA))
