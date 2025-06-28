import fitz  # PyMuPDF
import os
import requests
import streamlit as st
import time
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"
MIN_CHUNK_LENGTH = 30

# --- Step 1: Extract text and images from PDF pages ---
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

            img_filename = f"page_{i}_img_{img_index}.png"
            with open(img_filename, "wb") as img_file:
                img_file.write(image_bytes)
            images.append(img_filename)

        content_blocks.append({
            "page_num": i + 1,
            "text": text.strip(),
            "images": images
        })

    return content_blocks

# --- Step 2: Decide number of questions to generate ---
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

# --- Step 3: Split text into sentences and chunk it ---
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

    # Pad only with meaningful chunks
    while len(chunks) < parts and chunks:
        for chunk in chunks:
            if len(chunks) >= parts:
                break
            if len(chunk) >= min_length:
                chunks.append(chunk)

    return chunks[:parts] if chunks else ["[No meaningful content]"]

# --- Step 4: Chunk all pages for generating questions ---
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

    # Pad if needed using valid chunks
    while len(chunks) < num_questions and chunks:
        chunks.append(chunks[-1])

    if not chunks:
        return [{"text": "[No content extracted]", "images": []}]

    return chunks[:num_questions]

# --- Step 5: Query local LLM with a content chunk ---
def generate_question_local_llm(text, image_paths):
    image_prompt = f"Image paths: {image_paths}" if image_paths else "No images."
    prompt = f"""
You are an AI tutor. Based on the following educational content, generate a meaningful question that tests understanding:

Text:
{text}

{image_prompt}

The question should be based on the text or visual content.
Don't explain why the question is effective.
Just display the questions clearly.
After all the questions are displayed, show the answers below them.
Maintain the font well.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=90
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Question generation failed: {e}")
        return f"⚠️ [Error generating question: {str(e)}]"

# --- Step 6: Streamlit UI ---
st.title("📄 PDF Question Generator using Local LLM")
uploaded_pdf = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_pdf:
    temp_path = os.path.join("temp_" + uploaded_pdf.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_pdf.read())

    try:
        st.info("Extracting content from PDF...")
        content_blocks = extract_pdf_content(temp_path)
        num_pages = len(content_blocks)
        num_questions = get_question_count(num_pages)

        st.success(f"Total Pages: {num_pages} | Generating {num_questions} questions")

        chunks = chunk_pages(content_blocks, num_questions)

        for i, chunk in enumerate(chunks):
            st.write(f"\n**Generating Question {i+1}...**")
            print(f"[INFO] Q{i+1} | Text length: {len(chunk['text'])}, Images: {len(chunk['images'])}")

            if len(chunk["text"].strip()) < MIN_CHUNK_LENGTH:
                question = "⚠️ Skipped: Not enough content to generate a question."
            else:
                question = generate_question_local_llm(chunk["text"], chunk["images"])

            with st.expander(f"Q{i+1}: View Question and Answer"):
                st.write(question)

            time.sleep(1)  # Optional delay between requests

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        print(f"[ERROR] Exception: {e}")

    finally:
        # Cleanup temp files
        for block in content_blocks:
            for img_path in block["images"]:
                if os.path.exists(img_path):
                    os.remove(img_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

#https://colab.research.google.com/drive/1xp8Dhfk5O8FwwrY3YJKHsA31hmtzpoV9?usp=sharing
