import fitz  # PyMuPDF
import sys
import json

# Enforce UTF-8 encoding
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def extract_toc(pdf_path):
    with fitz.open(pdf_path) as doc:
        # Try to extract ToC if it exists
        toc = doc.get_toc()
        if toc:
            return toc

        # Custom extraction for ToC-like content
        nodes = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:  # Skip blocks without text lines
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Adjust threshold based on heading size in your PDF
                        if span["size"] > 12:
                            # Handle special characters safely
                            text = span["text"]
                            nodes.append(text.encode("utf-8", errors="ignore").decode("utf-8"))
        return nodes

# Example usage
pdf_path = r"D:\webscrap\HierarchialQA\textbook.pdf"
try:
    toc = extract_toc(pdf_path)
    print("Extracted ToC:", toc)
except Exception as e:
    print(f"Error extracting ToC: {e}")


import re


def merge_fragments(raw_toc):
    """
    Merge fragmented titles split across lines.
    """
    merged_toc = []
    buffer = ""

    for entry in raw_toc:
        if isinstance(entry, list):  # Nested list, keep as is
            merged_toc.append(entry)
            continue

        # Merge fragmented titles
        if not entry.endswith(".") and not re.match(r'^\d+(\.\d+)*$', entry):
            buffer += " " + entry.strip()
        else:
            if buffer:
                merged_toc.append(buffer.strip())
                buffer = ""
            merged_toc.append(entry.strip())

    if buffer:
        merged_toc.append(buffer.strip())

    return merged_toc

def parse_toc(merged_toc):
    """
    Parse merged ToC into a structured format like logic_toc.
    Dynamically detects chapters, sections, and subsections.
    """
    toc_dict = {}
    current_chapter = None
    current_section = None

    # Patterns for chapters, sections, and subsections
    chapter_pattern = re.compile(r'(Chapter \d+|Bibliography|Index)', re.IGNORECASE)
    section_pattern = re.compile(r'^\d+(\.\d+)*$')  # Numeric patterns like 1.1, 2.3

    for entry in merged_toc:
        if not isinstance(entry, str):
            continue

        # Match chapters
        if chapter_pattern.match(entry):
            current_chapter = entry
            toc_dict[current_chapter] = {}
            current_section = None  # Reset for new chapter

        # Match sections
        elif section_pattern.match(entry):
            if current_chapter:
                current_section = entry
                toc_dict[current_chapter][current_section] = []

        # Match subsections
        elif current_section and current_chapter:
            toc_dict[current_chapter][current_section].append(entry)

        # Fallback: Treat as a chapter if no chapter exists
        elif not current_chapter:
            current_chapter = entry
            toc_dict[current_chapter] = {}

    return toc_dict




try:
    # Step 1: Extract ToC
    raw_toc = extract_toc(pdf_path)

    # Step 2: Merge fragmented titles
    merged_toc = merge_fragments(raw_toc)
    print("\n\n\n")
    print("Merged ToC:", merged_toc)

    # Step 3: Parse ToC into desired format
    structured_toc = parse_toc(merged_toc)
    print("\n\n\n")
    print("str ToC:", structured_toc)


    # Step 4: Save structured ToC to JSON
    output_path = "structured_toc.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_toc, f, ensure_ascii=False, indent=4)
    print(f"Structured ToC saved to {output_path}")
except Exception as e:
    print(f"Error: {e}")