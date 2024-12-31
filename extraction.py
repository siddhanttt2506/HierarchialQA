import fitz 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        full_text += page.get_text()
    return full_text

def recursive_chunk_text(text, chunk_size=2000, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

def save_chunks_to_json(chunks, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

# List of PDF file paths
pdf_files = [
    r'D:\webscrap\HierarchialQA\nlp-book.pdf',
    r'D:\webscrap\HierarchialQA\LogicInCS.pdf',
    r'D:\webscrap\HierarchialQA\textbook.pdf'
]

# Directory to save the output JSON files
output_dir = r'D:\webscrap\HierarchialQA\output_chunks'
os.makedirs(output_dir, exist_ok=True)

# Process each PDF
for pdf_path in pdf_files:
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Chunk the extracted text
    text_chunks = recursive_chunk_text(extracted_text)
    
    # Define output JSON file path
    pdf_filename = os.path.basename(pdf_path)
    json_filename = os.path.splitext(pdf_filename)[0] + '_chunks.json'
    output_path = os.path.join(output_dir, json_filename)
    
    # Save chunks to JSON
    save_chunks_to_json(text_chunks, output_path)
    
    print(f'Processed {pdf_filename}, chunks saved to {output_path}')

    extracted_text = extract_text_from_pdf(pdf_path)
    print(f'Length of extracted text: {len(extracted_text)} characters')    




