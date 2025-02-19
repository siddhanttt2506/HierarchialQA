import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# Define a class to represent a node in the hierarchical tree
class Node:
    def __init__(self, node_id, title, level, content=None):
        self.node_id = node_id
        self.title = title
        self.level = level
        self.content = content
        self.summary = None
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def to_dict(self):
        return {
            "id": self.node_id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "children": [child.to_dict() for child in self.children]
        }


# Load models
embedding_model = SentenceTransformer('all-mpnet-base-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)


def assign_chunks_with_embeddings(subsection_title, chunks):
    """Find chunks relevant to the subsection title using semantic similarity."""
    subsection_embedding = embedding_model.encode(subsection_title)
    chunk_embeddings = {chunk: embedding_model.encode(chunk) for chunk in chunks}

    scores = {chunk: util.cos_sim(subsection_embedding, chunk_embedding).item() for chunk, chunk_embedding in chunk_embeddings.items()}
    # Select relevant chunks above a threshold
    threshold = 0.5  # Adjust as needed
    relevant_chunks = [chunk for chunk, score in scores.items() if score > threshold]

    # Sort by relevance and limit to top results
    relevant_chunks = sorted(relevant_chunks, key=lambda chunk: scores[chunk], reverse=True)[:8]
    return relevant_chunks


def generate_summary(text, max_length=150):
    """Generate a concise summary of the text."""
    if not text or len(text.strip()) < 50:  # Skip short texts
        return text

    try:
        max_chunk_length = 1024
        chunks = []

        if len(text) > max_chunk_length:
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

        summaries = []
        for chunk in chunks:
            chunk_words = len(chunk.split())
            max_summary_length = min(max_length, chunk_words)  # Ensure max_length doesn't exceed chunk length
            min_summary_length = max(10, max_summary_length - 10)  # Ensure min_length < max_length and >= 10

            if max_summary_length <= min_summary_length:
                summaries.append(chunk)  # Skip summarization for very short texts
            else:
                summary = summarizer(
                    chunk,
                    max_length=max_summary_length,
                    min_length=min_summary_length,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
        return " ".join(summaries)
    except Exception as e:
        print(f"Summarization error: {e}")
        return text[:max_length]


def build_hierarchical_tree(toc, chunks):
    """
    Build a hierarchical tree from the table of contents (ToC) and text chunks.
    """
    tree = Node("root", "Textbook", level=0)

    # Iterate through chapters in the ToC
    for chapter_idx, (chapter_title, sections) in enumerate(toc.items(), 1):
        chapter_id = f"chapter-{chapter_idx}"
        chapter_node = Node(chapter_id, chapter_title, level=1)

        for section_idx, (section_title, subsections) in enumerate(sections.items(), 1):
            section_id = f"{chapter_id}-section-{section_idx}"
            section_node = Node(section_id, section_title, level=2)

            # Process subsections
            for subsection_idx, subsection_title in enumerate(subsections, 1):
                subsection_id = f"{section_id}-subsection-{subsection_idx}"
                subsection_node = Node(subsection_id, subsection_title, level=3)

                # Find and assign relevant chunks
                relevant_chunks = assign_chunks_with_embeddings(subsection_title, chunks)
                subsection_node.content = "\n".join(relevant_chunks)

                # Generate summary for subsection
                if subsection_node.content:
                    subsection_node.summary = generate_summary(subsection_node.content)

                section_node.add_child(subsection_node)

            # Summarize section if it has subsections
            if section_node.children:
                combined_content = " ".join(child.content for child in section_node.children if child.content)
                section_node.content = combined_content
                section_node.summary = generate_summary(combined_content)

            chapter_node.add_child(section_node)

        # Summarize chapter if it has sections
        if chapter_node.children:
            combined_content = " ".join(child.content for child in chapter_node.children if child.content)
            chapter_node.content = combined_content
            chapter_node.summary = generate_summary(combined_content)

        tree.add_child(chapter_node)

    return tree


# Example ToC and chunk loading
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

ml_toc = {
    "1. Introduction": {
        "Why Machine Learning?": [
            "Introduction to Machine Learning",
            "Importance of ML",
            "ML Applications"
        ],
        "Problems Machine Learning Can Solve": [
            "Classification Problems",
            "Regression Problems",
            "Clustering Problems",
            "Common Use Cases"
        ],
        "Knowing Your Task and Knowing Your Data": [
            "Task Definition",
            "Data Understanding",
            "Data Requirements"
        ],
        "Why Python?": [
            "Python Features",
            "Python Libraries",
            "Python Community"
        ],
        "scikit-learn": [
            "Introduction to scikit-learn",
            "Key Features",
            "Basic Usage"
        ],
        "Installing scikit-learn": [
            "Installation Methods",
            "Dependencies",
            "Verification"
        ],
        "Essential Libraries and Tools": [
            "Jupyter Notebook",
            "NumPy",
            "SciPy",
            "matplotlib",
            "pandas",
            "mglearn"
        ],
        "Python 2 Versus Python 3": [
            "Key Differences",
            "Migration Considerations",
            "Best Practices"
        ],
        "Versions Used in this Book": [
            "Package Versions",
            "Compatibility Notes",
            "Version Management"
        ],
        "A First Application: Classifying Iris Species": [
            "Meet the Data",
            "Measuring Success: Training and Testing Data",
            "First Things First: Look at Your Data",
            "Building Your First Model: k-Nearest Neighbors",
            "Making Predictions",
            "Evaluating the Model"
        ],
        "Summary and Outlook": [
            "Chapter Review",
            "Key Takeaways",
            "Next Steps"
        ]
    },
    "2. Supervised Learning": {
        "Classification and Regression": [
            "Understanding Classification",
            "Understanding Regression",
            "Comparing Approaches"
        ],
        "Generalization, Overfitting, and Underfitting": [
            "Concept of Generalization",
            "Overfitting Problems",
            "Underfitting Problems",
            "Finding the Balance"
        ],
        "Relation of Model Complexity to Dataset Size": [
            "Dataset Size Impact",
            "Model Complexity Considerations",
            "Scaling Relationships"
        ],
        "Supervised Machine Learning Algorithms": [
            "Overview of Algorithms",
            "Algorithm Selection",
            "Performance Comparison"
        ],
        "Some Sample Datasets": [
            "Dataset Examples",
            "Dataset Characteristics",
            "Data Preparation"
        ],
        "k-Nearest Neighbors": [
            "KNN Algorithm",
            "KNN Parameters",
            "KNN Applications",
            "Implementation Details"
        ],
        "Linear Models": [
            "Linear Regression",
            "Logistic Regression",
            "Linear Classification",
            "Model Interpretation"
        ],
        "Naive Bayes Classifiers": [
            "Naive Bayes Concept",
            "Types of Naive Bayes",
            "Applications",
            "Implementation"
        ],
        "Decision Trees": [
            "Tree Structure",
            "Split Criteria",
            "Tree Parameters",
            "Pruning Techniques"
        ],
        "Ensembles of Decision Trees": [
            "Random Forests",
            "Gradient Boosting",
            "Ensemble Methods",
            "Performance Optimization"
        ],
        "Kernelized Support Vector Machines": [
            "Kernel Concept",
            "SVM Parameters",
            "Kernel Selection",
            "Implementation Details"
        ],
        "Neural Networks (Deep Learning)": [
            "Neural Network Basics",
            "Architectures",
            "Training Process",
            "Optimization Techniques"
        ],
        "Uncertainty Estimates from Classifiers": [
            "The Decision Function",
            "Predicting Probabilities",
            "Uncertainty in Multiclass Classification",
            "Confidence Metrics"
        ],
        "Summary and Outlook": [
            "Chapter Summary",
            "Key Learning Points",
            "Future Directions"
        ]
    },
    "3. Unsupervised Learning and Preprocessing": {
        "Types of Unsupervised Learning": [
            "Clustering Methods",
            "Dimensionality Reduction",
            "Density Estimation",
            "Pattern Discovery"
        ],
        "Challenges in Unsupervised Learning": [
            "Data Quality Issues",
            "Parameter Selection",
            "Evaluation Metrics",
            "Validation Approaches"
        ],
        "Preprocessing and Scaling": [
            "Different Kinds of Preprocessing",
            "Applying Data Transformations",
            "Scaling Training and Test Data the Same Way",
            "The Effect of Preprocessing on Supervised Learning"
        ],
        "Dimensionality Reduction, Feature Extraction, and Manifold Learning": [
            "Principal Component Analysis (PCA)",
            "Non-Negative Matrix Factorization (NMF)",
            "Manifold Learning with t-SNE",
            "Comparison of Methods"
        ],
        "Clustering": [
            "k-Means Clustering",
            "Agglomerative Clustering",
            "DBSCAN",
            "Comparing and Evaluating Clustering Algorithms",
            "Summary of Clustering Methods"
        ],
        "Summary and Outlook": [
            "Chapter Review",
            "Best Practices",
            "Advanced Topics"
        ]
    },
    "4. Representing Data and Engineering Features": {
        "Categorical Variables": [
            "Types of Categories",
            "Encoding Methods",
            "Handling Missing Values"
        ],
        "One-Hot-Encoding (Dummy Variables)": [
            "Encoding Process",
            "Handling Missing Values",
            "Implementation Details"
        ],
        "Numbers Can Encode Categoricals": [
            "Ordinal Encoding",
            "Label Encoding",
            "Best Practices"
        ],
        "Binning, Discretization, Linear Models, and Trees": [
            "Binning Methods",
            "Discretization Techniques",
            "Impact on Models"
        ],
        "Interactions and Polynomials": [
            "Feature Interactions",
            "Polynomial Features",
            "Feature Selection"
        ],
        "Univariate Nonlinear Transformations": [
            "Log Transform",
            "Power Transform",
            "Other Transformations"
        ],
        "Automatic Feature Selection": [
            "Univariate Statistics",
            "Model-Based Feature Selection",
            "Iterative Feature Selection"
        ],
        "Utilizing Expert Knowledge": [
            "Domain Expertise",
            "Feature Engineering",
            "Best Practices"
        ],
        "Summary and Outlook": [
            "Key Concepts",
            "Advanced Topics",
            "Future Directions"
        ]
    },
    "5. Model Evaluation and Improvement": {
        "Cross-Validation": [
            "Cross-Validation in scikit-learn",
            "Benefits of Cross-Validation",
            "Stratified k-Fold Cross-Validation and Other Strategies"
        ],
        "Grid Search": [
            "Simple Grid Search",
            "The Danger of Overfitting the Parameters and the Validation Set",
            "Grid Search with Cross-Validation"
        ],
        "Evaluation Metrics and Scoring": [
            "Keep the End Goal in Mind",
            "Metrics for Binary Classification",
            "Metrics for Multiclass Classification",
            "Regression Metrics",
            "Using Evaluation Metrics in Model Selection"
        ],
        "Summary and Outlook": [
            "Chapter Summary",
            "Best Practices",
            "Next Steps"
        ]
    },
    "6. Algorithm Chains and Pipelines": {
        "Parameter Selection with Preprocessing": [
            "Parameter Types",
            "Selection Methods",
            "Optimization Strategies"
        ],
        "Building Pipelines": [
            "Pipeline Components",
            "Pipeline Construction",
            "Error Handling"
        ],
        "Using Pipelines in Grid Searches": [
            "Search Strategy",
            "Parameter Grid",
            "Optimization Methods"
        ],
        "The General Pipeline Interface": [
            "Interface Components",
            "Usage Patterns",
            "Best Practices"
        ],
        "Convenient Pipeline Creation with make_pipeline": [
            "Creation Methods",
            "Pipeline Options",
            "Common Patterns"
        ],
        "Accessing Step Attributes": [
            "Attribute Types",
            "Access Methods",
            "Debugging Tips"
        ],
        "Accessing Attributes in a Grid-Searched Pipeline": [
            "Search Results",
            "Attribute Access",
            "Result Analysis"
        ],
        "Grid-Searching Preprocessing Steps and Model Parameters": [
            "Search Space",
            "Optimization",
            "Performance Tuning"
        ],
        "Grid-Searching Which Model To Use": [
            "Model Selection",
            "Comparison Metrics",
            "Decision Criteria"
        ],
        "Summary and Outlook": [
            "Key Learnings",
            "Advanced Topics",
            "Future Developments"
        ]
    },
    "7. Working with Text Data": {
        "Types of Data Represented as Strings": [
            "Text Types",
            "String Representations",
            "Processing Methods"
        ],
        "Example Application: Sentiment Analysis of Movie Reviews": [
            "Representing Text Data as a Bag of Words",
            "Applying Bag-of-Words to a Toy Dataset",
            "Bag-of-Words for Movie Reviews",
            "Stopwords",
            "Rescaling the Data with tf–idf",
            "Investigating Model Coefficients",
            "Bag-of-Words with More Than One Word (n-Grams)",
            "Advanced Tokenization, Stemming, and Lemmatization"
        ],
        "Topic Modeling and Document Clustering": [
            "Latent Dirichlet Allocation",
            "Document Clustering",
            "Evaluation Methods"
        ],
        "Summary and Outlook": [
            "Chapter Review",
            "Advanced Concepts",
            "Future Directions"
        ]
    },
    "8. Wrapping Up": {
        "Approaching a Machine Learning Problem": [
            "Problem Definition",
            "Solution Strategy",
            "Implementation Steps"
        ],
        "Humans in the Loop": [
            "Human Interaction",
            "Feedback Integration",
            "System Design"
        ],
        "From Prototype to Production": [
            "Testing Production Systems",
            "Building Your Own Estimator",
            "Deployment Strategies"
        ],
        "Where to Go from Here": [
            "Theory",
            "Other Machine Learning Frameworks and Packages",
            "Ranking, Recommender Systems, and Other Kinds of Learning",
            "Probabilistic Modeling, Inference, and Probabilistic Programming",
            "Neural Networks",
            "Scaling to Larger Datasets",
            "Honing Your Skills"
        ],
        "Conclusion": [
            "Final Thoughts",
            "Key Takeaways",
            "Next Steps"
        ]
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


# Load chunks from a JSON file
with open(r"D:\webscrap\HierarchialQA\output_chunks\mlbasics_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Build hierarchical tree
hierarchical_tree = build_hierarchical_tree(ml_toc, chunks)

# Save hierarchical tree to JSON file
output_path = "hierarchical_tree_with_summaries_ml.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(hierarchical_tree.to_dict(), f, ensure_ascii=False, indent=4)

print(f"Hierarchical tree saved to {output_path}")
