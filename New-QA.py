import fitz  # PyMuPDF
import os
import requests
import re
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"
MIN_CHUNK_LENGTH = 30

# --- Extract PDF content: text + images ---
def extract_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)
    content_blocks = []

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text()
        images = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_filename = f"{os.path.splitext(pdf_path)[0]}_page_{i}_img_{img_index}.png"

            with open(img_filename, "wb") as img_file:
                img_file.write(image_bytes)

            images.append(img_filename)

        content_blocks.append({
            "page_num": i + 1,
            "text": text.strip(),
            "images": images
        })

    return content_blocks

# --- Question count logic based on PDF length ---
def get_question_count(num_pages):
    if num_pages <= 8:
        return num_pages * 2
    elif num_pages <= 15:
        return num_pages
    elif num_pages <= 49:
        return num_pages // 2
    elif num_pages <= 99:
        return num_pages // 5
    else:
        return min(num_pages // 5, 20)

# --- Split text into chunks ---
def split_text_by_sentences(text, parts, min_length=MIN_CHUNK_LENGTH):
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    total_sentences = len(sentences)
    chunks = []
    if total_sentences == 0:
        return ["[No content]"]

    sentences_per_chunk = max(1, total_sentences // parts)

    for i in range(0, total_sentences, sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if len(chunk.strip()) >= min_length:
            chunks.append(chunk.strip())

    while len(chunks) < parts and chunks:
        for chunk in chunks:
            if len(chunks) >= parts:
                break
            if len(chunk) >= min_length:
                chunks.append(chunk)

    return chunks[:parts] if chunks else ["[No meaningful content]"]

# --- Chunk pages based on desired number of questions ---
def chunk_pages(content_blocks, num_questions):
    chunks = []
    total_pages = len(content_blocks)
    questions_per_page = max(1, num_questions // total_pages)

    for block in content_blocks:
        page_text = block["text"]
        page_images = block["images"]

        if page_text:
            split_chunks = split_text_by_sentences(page_text, questions_per_page)
            for part_text in split_chunks:
                if len(part_text.strip()) >= MIN_CHUNK_LENGTH:
                    chunks.append({
                        "text": part_text,
                        "images": page_images
                    })

    while len(chunks) < num_questions and chunks:
        chunks.append(chunks[-1])

    return chunks[:num_questions] if chunks else [{"text": "[No content extracted]", "images": []}]

# --- Generate higher-order Q&A using local LLM ---
def generate_question_local_llm(text, image_paths):
    prompt = f"""
You are an AI tutor. Your task is to generate **high-quality, challenging questions** that require reasoning, comparison, or conceptual understanding. Avoid simple fact-based or one-line questions.

Given the following content, create questions that:
 Involve comparison (e.g., differences between data points or ideas),Ask "why" or "how" something works, Involve interpretation of visual data (tables, charts, images) 

Input Text:
{text}
    
{"Attached images: " + ", ".join(image_paths) if image_paths else "No visual content."}

Instructions:
- DO NOT ask simple recall questions.
- Prefer questions like "What does this imply?", "How does X differ from Y?", or "Why is X significant in this context?"
- If content includes a table or image, analyze it to generate a more meaningful question than just asking for a value.
- After all questions, provide clear, correct answers.
- Just output the formatted questions followed by answers (no extra explanation or reasoning on how you built the question).

Output format:
Q1: [Question]
A1: [Answer]
Q2: ...
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.RequestException as e:
        return f"[Error] Failed to generate: {str(e)}"

# --- Save Q&A as a TXT file ---
def save_qna_to_txt(pdf_path, qa_text):
    filename = os.path.splitext(pdf_path)[0] + "_QnA.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Questions & Answers for {os.path.basename(pdf_path)}\n\n")
        f.write(qa_text.strip())
    return filename

# --- Main processor for a single PDF ---
def process_single_pdf(pdf_path):
    try:
        content_blocks = extract_pdf_content(pdf_path)
        num_pages = len(content_blocks)
        num_questions = get_question_count(num_pages)
        chunks = chunk_pages(content_blocks, num_questions)

        all_qna_output = []

        for i, chunk in enumerate(chunks):
            if len(chunk["text"].strip()) < MIN_CHUNK_LENGTH:
                qna = f"Q{i+1}: Skipped (Not enough content)\n"
            else:
                qna = generate_question_local_llm(chunk["text"], chunk["images"])
                qna = f"{qna}\n"
            all_qna_output.append(qna)

        combined_text = "\n\n".join(all_qna_output)
        txt_file = save_qna_to_txt(pdf_path, combined_text)
        print(f"[✅] Saved: {txt_file}")
        return txt_file

    finally:
        # Clean up temporary images
        for block in content_blocks:
            for img_path in block["images"]:
                if os.path.exists(img_path):
                    os.remove(img_path)

# --- Streamlit UI ---
st.title("📄 Multi-PDF High-Quality QnA Generator (Local LLM)")
folder_path = st.text_input("📁 Enter the folder path containing PDFs")

if folder_path and os.path.isdir(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        st.warning("⚠️ No PDF files found in this folder.")
    else:
        st.success(f"Found {len(pdf_files)} PDF(s). Starting generation...")

        for pdf_file in pdf_files:
            full_path = os.path.join(folder_path, pdf_file)
            st.write(f"🔍 Processing: `{pdf_file}`")

            try:
                output_file = process_single_pdf(full_path)
                st.success(f"✅ QnA TXT saved: `{os.path.basename(output_file)}`")
            except Exception as e:
                st.error(f"❌ Failed to process `{pdf_file}`: {str(e)}")
else:
    st.info("🔎 Please enter a valid folder path with PDF files.")
#pip install streamlit pymupdf requests
