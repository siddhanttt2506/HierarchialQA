import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# Define a class to represent a node in the hierarchical tree
class Node:
    def __init__(self, node_id, title, content=None):
        self.node_id = node_id
        self.title = title
        self.content = content
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def to_dict(self):
        return {
            "id": self.node_id,
            "title": self.title,
            "content": self.content,
            "children": [child.to_dict() for child in self.children]
        }

# Load a pre-trained model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="path_to_cache")

def assign_chunks_with_embeddings(subsection_title, chunks):
    """Find chunks relevant to the subsection title using semantic similarity."""
    subsection_embedding = model.encode(subsection_title)
    chunk_embeddings = {chunk: model.encode(chunk) for chunk in chunks}
    threshold = 0.5 
    scores = {chunk: util.cos_sim(subsection_embedding, chunk_embedding).item() for chunk, chunk_embedding in chunk_embeddings.items()}
    # Select top 5 most relevant chunks
    relevant_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:6]
    return [chunk for chunk, score in relevant_chunks]

# Function to parse the table of contents and build the hierarchical tree
def build_hierarchical_tree(toc, chunks):
    """
    Build a hierarchical tree from the table of contents (ToC) and text chunks.
    """
    tree = Node("root", "Textbook")
    chapter_nodes = {}  # Track chapters by ID

    # Iterate through chapters in the ToC
    for chapter_title, sections in toc.items():
        chapter_id = f"chapter-{len(chapter_nodes) + 1}"
        chapter_node = Node(chapter_id, chapter_title)
        chapter_nodes[chapter_id] = chapter_node  # Store chapter for tracking

        # Process sections in the chapter
        for section_title, subsections in sections.items():
            section_id = f"{chapter_id}-section-{len(chapter_node.children) + 1}"
            section_node = Node(section_id, section_title)

            # Process subsections
            for subsection_title in subsections:
                subsection_id = f"{section_id}-subsection-{len(section_node.children) + 1}"
                subsection_node = Node(subsection_id, subsection_title)

                # Find and assign relevant chunks
                relevant_chunks = find_relevant_chunks(subsection_title, chunks)
                subsection_node.content = "\n".join(relevant_chunks)
                section_node.add_child(subsection_node)

            # Add section to chapter
            chapter_node.add_child(section_node)

        # Add chapter to the root node
        tree.add_child(chapter_node)

    return tree


# Example ToC (table of contents) and chunk loading
textbook_toc = {
    "Linear Equations": {
        "Fields": [],
        "Systems of Linear Equations": [],
        "Matrices and Elementary Row Operations": [],
        "Row-Reduced Echelon Matrices": [],
        "Matrix Multiplication": [],
        "Invertible Matrices": []
    },
    "Vector Spaces": {
        "Vector Spaces": [],
        "Subspaces": [],
        "Bases and Dimension": [],
        "Coordinates": [],
        "Summary of Row-Equivalence": [],
        "Computations Concerning Subspaces": []
    },
    "Linear Transformations": {
        "Linear Transformations": [],
        "The Algebra of Linear Transformations": [],
        "Isomorphism": [],
        "Representation of Transformations by Matrices": [],
        "Linear Functionals": [],
        "The Double Dual": [],
        "The Transpose of a Linear Transformation": []
    },
    "Polynomials": {
        "Algebras": [],
        "The Algebra of Polynomials": [],
        "Lagrange Interpolation": [],
        "Polynomial Ideals": [],
        "The Prime Factorization of a Polynomial": []
    },
    "Determinants": {
        "Commutative Rings": [],
        "Determinant Functions": [],
        "Permutations and the Uniqueness of Determinants": [],
        "Additional Properties of Determinants": [],
        "Modules": [],
        "Multilinear Functions": [],
        "The Grassman Ring": []
    },
    "Elementary Canonical Forms": {
        "Introduction": [],
        "Characteristic Values": [],
        "Annihilating Polynomials": [],
        "Invariant Subspaces": [],
        "Simultaneous Triangulation; Simultaneous Diagonalization": [],
        "Direct-Sum Decompositions": [],
        "Invariant Direct Sums": [],
        "The Primary Decomposition Theorem": []
    },
    "The Rational and Jordan Forms": {
        "Cyclic Subspaces and Annihilators": [],
        "Cyclic Decompositions and the Rational Form": [],
        "The Jordan Form": [],
        "Computation of Invariant Factors": [],
        "Summary; Semi-Simple Operators": []
    },
    "Inner Product Spaces": {
        "Inner Products": [],
        "Inner Product Spaces": [],
        "Linear Functionals and Adjoints": [],
        "Unitary Operators": [],
        "Normal Operators": []
    },
    "Operators on Inner Product Spaces": {
        "Introduction": [],
        "Forms on Inner Product Spaces": [],
        "Positive Forms": [],
        "More on Forms": [],
        "Spectral Theory": [],
        "Further Properties of Normal Operators": []
    },
    "Bilinear Forms": {
        "Bilinear Forms": [],
        "Symmetric Bilinear Forms": [],
        "Skew-Symmetric Bilinear Forms": [],
        "Groups Preserving Bilinear Forms": []
    },
    "Appendix": {
        "Sets": [],
        "Functions": [],
        "Equivalence Relations": [],
        "Quotient Spaces": [],
        "Equivalence Relations in Linear Algebra": [],
        "The Axiom of Choice": []
    }
}

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
nlp_toc = { 
    "The Information Age": {
        "The Information Age": [],
        "Technology for Accessing Info": [],
        "Question Answering Systems": [
            "ELIZA - The Rogerian Therapist",
            "Early NLP Systems",
            "Foundations of Story Understanding",
            "In-Depth Understanding",
            "Turing Test"
        ],
        "Information Retrieval": [
            "IR Defined",
            "Documents as Bags-of-Words",
            "The Vector Space Model",
            "Performance Evaluation",
            "Measuring Relevance",
            "Challenges in Information Retrieval"
        ],
        "Information Extraction": [
            "What is Information Extraction?",
            "Information Extraction Tasks",
            "Architecture of an IE System"
        ],
        "Automatic Summarization": [
            "Why Summarization?",
            "Approaches to Automatic Summarization",
            "Summarization in Relation to Information Extraction",
            "Summarization in Relation to Other Technologies",
            "Evaluation of Summarization Systems",
            "Summarization in the Context of Indian Tradition"
        ],
        "Automatic Text Categorization": [
            "Why Text Categorization?",
            "Approaches to Automatic Text Categorization",
            "Text Representation",
            "Feature Weighting",
            "Text Classification and Clustering"
        ],
        "Machine Translation": [
            "Machine Translation is Hard",
            "Deploying Machine Translation",
            "Approaches to Machine Translation",
            "Challenges in Machine Translation",
            "Machine Translation in India"
        ],
        "Speech Technologies": [
            "Automatic Speech Recognition",
            "Speech Synthesis",
            "Other Speech Technologies"
        ],
        "Human and Machine Intelligence": [],
        "Shape of Things to Come": []
    },
    "Foundations of NLP": {
        "Introduction": [
            "Language, Communication, Technology",
            "Natural Language Processing and Computational Linguistics",
            "NLP: An AI Perspective",
            "NLP Over the Decades",
            "Linguistics versus NLP"
        ],
        "Computational Linguistics": [
            "Dictionaries",
            "Thesauri and WordNets",
            "Morphology",
            "POS Tagging",
            "Syntax: Grammars and Parsers",
            "Semantics",
            "Pragmatics",
            "Other Areas of Linguistics"
        ],
        "Statistical Approaches": [
            "Corpora",
            "Statistical Approaches to Language",
            "Machine Learning"
        ],
        "Indian Language Technologies": [
            "The Text Processing Environment",
            "The Alphabet",
            "The Script Grammar",
            "Fonts, Glyphs and Encoding Standards",
            "Character Encoding Standards",
            "Romanization",
            "Spell Checkers",
            "Optical Character Recognition",
            "Language Identification",
            "Others Technologies for Indian Languages",
            "NLP and Sanskrit",
            "Epilogue"
        ],
        "Conclusions": []
    },
    "Advances in IR": {
        "History of IR": [
            "From The Library to the Internet"
        ],
        "Basic IR Models": [
            "IR Models",
            "Term Weighting: tf-idf",
            "Similarity Measures",
            "The Probability Ranking Principle",
            "Performance Evaluation"
        ],
        "Towards Intelligent IR": [
            "Improving User Queries - Relevance Feedback",
            "Page Ranking",
            "Role of Linguistics",
            "Latent Semantic Indexing",
            "Meta Search Engines",
            "Semantic Web",
            "Information Retrieval is Difficult",
            "Conclusions"
        ]
    },
    "Appendices": {
        "C5 Tag Set": [],
        "Sample Sentences": [],
        "ISCII Character Set": []
    }
}


# Load the chunked content from a JSON file (replace this with the actual file path)
with open(r"D:\webscrap\HierarchialQA\output_chunks\LogicInCS_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Build the hierarchical tree
hierarchical_tree = build_hierarchical_tree(logic_toc, chunks)

# Save the hierarchical tree to a JSON file
output_path = "hierarchical_tree.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(hierarchical_tree.to_dict(), f, ensure_ascii=False, indent=4)

print(f"Hierarchical tree saved to {output_path}")