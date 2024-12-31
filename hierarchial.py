import json
from collections import defaultdict

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

# Function to parse the table of contents and build the hierarchical tree
def build_hierarchical_tree(toc, chunks):
    tree = Node("root", "Textbook")
    chapter_nodes = {}

    # Create chapters, sections, and subsections from the table of contents
    for chapter_title, sections in toc.items():
        chapter_id = f"chapter-{len(chapter_nodes) + 1}"
        chapter_node = Node(chapter_id, chapter_title)

        for section_title, subsections in sections.items():
            section_id = f"{chapter_id}-section-{len(chapter_node.children) + 1}"
            section_node = Node(section_id, section_title)

            for subsection_title in subsections:
                subsection_id = f"{section_id}-subsection-{len(section_node.children) + 1}"
                subsection_node = Node(subsection_id, subsection_title)

                # Assign text chunks to the leaf nodes (subsections)
                relevant_chunks = [chunk for chunk in chunks if subsection_title in chunk]
                subsection_node.content = "\n".join(relevant_chunks)
                section_node.add_child(subsection_node)

            chapter_node.add_child(section_node)

        tree.add_child(chapter_node)

    return tree

# Example ToC (table of contents) and chunk loading
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

# Load the chunked content from a JSON file (replace this with the actual file path)
with open("D:\webscrap\HierarchialQA\output_chunks\LogicInCS_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Build the hierarchical tree
hierarchical_tree = build_hierarchical_tree(logic_toc, chunks)

# Save the hierarchical tree to a JSON file
output_path = "hierarchical_tree.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(hierarchical_tree.to_dict(), f, ensure_ascii=False, indent=4)

print(f"Hierarchical tree saved to {output_path}")