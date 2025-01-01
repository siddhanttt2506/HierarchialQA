import json
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from transformers import pipeline
import torch

@dataclass
class ContentNode:
    node_id: str
    title: str
    level: int
    content: Optional[str] = None
    summary: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    children: List['ContentNode'] = field(default_factory=list)
    parent: Optional['ContentNode'] = None
    
    def to_dict(self):
        return {
            "id": self.node_id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "children": [child.to_dict() for child in self.children]
        }

class EnhancedBookProcessor:
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 cache_folder: str = "./model_cache",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.embedding_model = SentenceTransformer(embedding_model, cache_folder=cache_folder)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        self.embedding_cache = {}
        self.chunk_embeddings = {}
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get or compute embedding for text with caching."""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.embedding_model.encode(text, convert_to_tensor=True)
        return self.embedding_cache[text]

    def preprocess_chunks(self, chunks: List[str]):
        """Precompute embeddings for all chunks."""
        print("Precomputing chunk embeddings...")
        texts = list(set(chunks))  # Remove duplicates
        embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
        self.chunk_embeddings = {text: emb for text, emb in zip(texts, embeddings)}
        
    def find_relevant_chunks(self, 
                           query: str, 
                           chunks: List[str], 
                           threshold: float = 0.5,
                           top_k: int = 6) -> List[str]:
        """Find most relevant chunks using pre-computed embeddings."""
        query_embedding = self.get_embedding(query)
        
        # Calculate similarities using pre-computed embeddings
        similarities = []
        for chunk in chunks:
            if chunk in self.chunk_embeddings:
                score = util.cos_sim(query_embedding, self.chunk_embeddings[chunk]).item()
                if score > threshold:
                    similarities.append((chunk, score))
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in similarities[:top_k]]

    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """Generate a concise summary of the text."""
        if not text or len(text.strip()) < 50:  # Skip very short texts
            return text
            
        try:
            # Split text into chunks if it's too long
            max_chunk_length = 1024
            chunks = []
            
            if len(text) > max_chunk_length:
                # Split into sentences roughly
                sentences = text.split(". ")
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_length:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk)
            else:
                chunks = [text]
            
            # Process each chunk
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                    
                # Calculate appropriate length constraints
                chunk_words = len(chunk.split())
                max_summary_length = min(max_length, chunk_words)
                min_summary_length = min(30, max_summary_length - 1)  # Ensure min is less than max
                
                if max_summary_length <= min_summary_length:
                    summaries.append(chunk)
                else:
                    try:
                        summary = self.summarizer(
                            chunk,
                            max_length=max_summary_length,
                            min_length=min_summary_length,
                            do_sample=False
                        )[0]['summary_text']
                        summaries.append(summary)
                    except Exception as e:
                        print(f"Chunk summarization failed: {e}")
                        summaries.append(chunk[:max_summary_length])
            
            return " ".join(summaries)
        except Exception as e:
            print(f"Warning: Summarization failed: {e}")
            return text[:max_length]

    def build_hierarchical_tree(self, toc: Dict, chunks: List[str]) -> ContentNode:
        """Build enhanced hierarchical tree with summaries at all levels."""
        # Precompute all chunk embeddings
        self.preprocess_chunks(chunks)
        
        root = ContentNode(node_id="root", title="Textbook", level=0)
        
        def process_level(parent: ContentNode, content: Dict, level: int, path: str = ""):
            for idx, (title, subcontents) in enumerate(content.items(), 1):
                node_id = f"{path}_{idx}" if path else f"chapter_{idx}"
                current_node = ContentNode(
                    node_id=node_id,
                    title=title,
                    level=level,
                    parent=parent
                )
                
                # If it's a subsection (leaf node)
                if not subcontents:
                    relevant_chunks = self.find_relevant_chunks(title, chunks)
                    current_node.content = "\n".join(relevant_chunks)
                    if current_node.content:
                        current_node.summary = self.generate_summary(current_node.content)
                
                # Process deeper levels
                if isinstance(subcontents, dict):
                    process_level(current_node, subcontents, level + 1, node_id)
                    
                    # Generate summary for non-leaf nodes from children's content
                    if current_node.children:
                        child_content = " ".join(child.content for child in current_node.children if child.content)
                        if child_content:
                            current_node.summary = self.generate_summary(child_content)
                
                parent.children.append(current_node)
        
        process_level(root, toc, 1)
        return root

def process_book(toc_file: str, chunks_file: str, output_file: str):
    """Process a book and save the hierarchical structure."""
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Initialize processor
    processor = EnhancedBookProcessor()
    
    # Build tree
    print("Building hierarchical tree...")
    tree = processor.build_hierarchical_tree(toc_file, chunks)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tree.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"Processed book saved to {output_file}")
    return tree

logic_toc = {
    "Propositional Logic": {
        "Declarative Sentences": [],
        "Natural Deduction": ["Rules for Natural Deduction", "Derived Rules", "Natural Deduction in Summary", "Provable Equivalence", "Proof by Contradiction"],
        "Semantics of Propositional Logic": ["Meaning of Logical Connectives", "Mathematical Induction", "Soundness of Propositional Logic", "Completeness of Propositional Logic"],
        "Normal Forms": ["Semantic Equivalence, Satisfiability and Validity", "Conjunctive Normal Forms and Validity", "Horn Clauses and Satisfiability"],
        "SAT Solvers": ["A Linear Solver", "A Cubic Solver"]
    },
    "Predicate Logic": {
        "The Need for a Richer Language": [],
        "Predicate Logic as a Formal Language": ["Terms", "Formulas", "Free and Bound Variables", "Substitution"],
        "Proof Theory of Predicate Logic": ["Natural Deduction Rules", "Quantifier Equivalences"],
        "Semantics of Predicate Logic": ["Models", "Semantic Entailment", "The Semantics of Equality"],
        "Undecidability of Predicate Logic": [],
        "Expressiveness of Predicate Logic": ["Existential Second-Order Logic", "Universal Second-Order Logic"],
        "Micromodels of Software": ["State Machines", "Alma – Revisited", "A Software Micromodel"]
    },
    "Verification by Model Checking": {
        "Motivation for Verification": [],
        "Linear-Time Temporal Logic": ["Syntax of LTL", "Semantics of LTL", "Practical Patterns of Specifications", "Important Equivalences Between LTL Formulas", "Adequate Sets of Connectives for LTL"],
        "Model Checking: Systems, Tools, Properties": ["Example: Mutual Exclusion", "The NuSMV Model Checker", "Running NuSMV", "Mutual Exclusion Revisited", "The Ferryman", "The Alternating Bit Protocol"],
        "Branching-Time Logic": ["Syntax of CTL", "Semantics of CTL", "Practical Patterns of Specifications", "Important Equivalences Between CTL Formulas", "Adequate Sets of CTL Connectives"],
        "CTL* and the Expressive Powers of LTL and CTL": ["Boolean Combinations of Temporal Formulas in CTL", "Past Operators in LTL"],
        "Model-Checking Algorithms": ["The CTL Model-Checking Algorithm", "CTL Model Checking with Fairness", "The LTL Model-Checking Algorithm"],
        "The Fixed-Point Characterisation of CTL": ["Monotone Functions", "The Correctness of SATEG", "The Correctness of SATEU"]
    },
    "Program Verification": {
        "Why Should We Specify and Verify Code?": [],
        "A Framework for Software Verification": ["A Core Programming Language", "Hoare Triples", "Partial and Total Correctness", "Program Variables and Logical Variables"],
        "Proof Calculus for Partial Correctness": ["Proof Rules", "Proof Tableaux", "A Case Study: Minimal-Sum Section"],
        "Proof Calculus for Total Correctness": [],
        "Programming by Contract": []
    },
    "Modal Logics and Agents": {
        "Modes of Truth": [],
        "Basic Modal Logic": ["Syntax", "Semantics"],
        "Logic Engineering": ["The Stock of Valid Formulas", "Important Properties of the Accessibility Relation", "Correspondence Theory", "Some Modal Logics"],
        "Natural Deduction": [],
        "Reasoning About Knowledge in a Multi-Agent System": ["Some Examples", "The Modal Logic KT45n", "Natural Deduction for KT45n", "Formalising the Examples"]
    },
    "Binary Decision Diagrams": {
        "Representing Boolean Functions": ["Propositional Formulas and Truth Tables", "Binary Decision Diagrams", "Ordered BDDs"],
        "Algorithms for Reduced OBDDs": ["The Algorithm Reduce", "The Algorithm Apply", "The Algorithm Restrict", "The Algorithm Exists"],
        "Symbolic Model Checking": ["Representing Subsets of the Set of States", "Representing the Transition Relation", "Implementing the Functions pre∃ and pre∀", "Synthesising OBDDs"],
        "A Relational Mu-Calculus": ["Syntax and Semantics", "Coding CTL Models and Specifications"]
    }
}

# Example usage
toc = logic_toc  # Your existing TOC
with open(r"D:\webscrap\HierarchialQA\output_chunks\LogicInCS_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
output_file = "enhanced_hierarchical_tree.json"

# Process the book
processor = EnhancedBookProcessor()
tree = processor.build_hierarchical_tree(toc, chunks)

# Save the results
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(tree.to_dict(), f, ensure_ascii=False, indent=2)