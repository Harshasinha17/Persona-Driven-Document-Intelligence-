import os
import json
import datetime
import pdfplumber
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

# Load inputs
DATA_DIR = "../data"
OUTPUT_DIR = "../output"

model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Load Persona and Job
with open(os.path.join(DATA_DIR, "persona.txt")) as f:
    persona = f.read().strip()

with open(os.path.join(DATA_DIR, "job.txt")) as f:
    job = f.read().strip()

job_embedding = model.encode(job)

# Step 2: Process PDFs
pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
extracted_sections = []
subsection_analysis = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(DATA_DIR, pdf_file)
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4].strip()
            if len(text) < 20:  # skip short lines
                continue

            score = util.cos_sim(model.encode(text), job_embedding).item()

            if score > 0.5:
                extracted_sections.append({
                    "document": pdf_file,
                    "section_title": text.split("\n")[0][:100],
                    "importance_rank": 0,  # we'll fill ranks later
                    "page_number": page_num
                })
                subsection_analysis.append({
                    "document": pdf_file,
                    "refined_text": text,
                    "page_number": page_num
                })

# Step 3: Sort by relevance
extracted_sections = sorted(extracted_sections, key=lambda x: -model.encode(x["section_title"]).dot(job_embedding))
for i, section in enumerate(extracted_sections):
    section["importance_rank"] = i + 1

# Step 4: Final Output
output = {
    "metadata": {
        "input_documents": pdf_files,
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.datetime.now().isoformat()
    },
    "extracted_sections": extracted_sections,
    "subsection_analysis": subsection_analysis
}

# Save JSON
with open(os.path.join(OUTPUT_DIR, "result.json"), "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print("✅ Output generated successfully.")
