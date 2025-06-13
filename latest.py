import logging
import time
from pathlib import Path
import pandas as pd
import ollama
from docling.document_converter import DocumentConverter
import fitz  # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import os

# Setup logger
_log = logging.getLogger(__name__)

# ---------------------------- OLLAMA CONFIG ---------------------------- #
OLLAMA_MODEL_NAME = 'gemma3:4b'       # Text model for table descriptions
OLLAMA_IMAGE_MODEL = 'gemma3:4b'          # Vision model for image descriptions

# ---------------------------- TABLE DESCRIPTION ---------------------------- #
def generate_table_description(table_df: pd.DataFrame) -> str:
    _log.info("Generating description with local Ollama LLM...")

    table_markdown = table_df.to_markdown(index=False)

    prompt = f"""
You are a professional data analyst. Your job is to write a clear, concise, and insightful summary of the table provided. Your explanation should be easy for a general audience to understand, while still offering a complete understanding of the table's purpose and key details. Follow these instructions carefully:
1. Explain the Table’s Purpose (1 sentence)
2. Highlight Key Data Points or Trends
3. Interpret Financial/Numerical Data Clearly
4. Describe with Full Context
5. Extract maximum information from the table and give me detailed description.
6. At the end don't ask back any questions.
Here is the table data:
---
{table_markdown}
---
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        _log.error(f"Error communicating with Ollama LLM: {e}")
        return "Error: Could not generate a description using the LLM."


# ---------------------------- IMAGE DESCRIPTION ---------------------------- #
def generate_image_description(image_path: Path) -> str:
    _log.info(f"Describing image: {image_path.name}")

    prompt = (
        "Please describe this image in detail. "
        "Identify any text, structure, layout, or objects that may appear. "
        "This is for document analysis, so include tables, diagrams, charts, or figures if visible."
        "All the information from the image must be extracted properly"
        "Observe the patterns in graphs and explain them and what they are depicting."
        "At the end don't ask back any questions"
    )

    try:
        response = ollama.chat(
            model=OLLAMA_IMAGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path.resolve())]
                }
            ]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        _log.error(f"Error during image description: {e}")
        return "Error: Could not generate image description."


# ---------------------------- TABLE EXTRACTION ---------------------------- #
def extract_tables(pdf_path: Path) -> dict:
    results = {
        'tables': [],
        'output_files': []
    }

    output_dir = Path("extracted_tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename_stem = pdf_path.stem

    try:
        doc_converter = DocumentConverter()
        conv_res = doc_converter.convert(pdf_path)
        _log.info(f"Found {len(conv_res.document.tables)} tables in the document.")

        for table_ix, table in enumerate(conv_res.document.tables):
            table_number = table_ix + 1
            table_df = table.export_to_dataframe()
            description = generate_table_description(table_df)

            base_name = f"{doc_filename_stem}-table-{table_number}"
            csv_path = output_dir / f"{base_name}.csv"
            html_path = output_dir / f"{base_name}.html"
            desc_path = output_dir / f"{base_name}-description.txt"

            table_df.to_csv(csv_path, index=False)
            with html_path.open("w", encoding="utf-8") as fp:
                fp.write(table.export_to_html(doc=conv_res.document))
            with desc_path.open("w") as f:
                f.write(description)

            results['tables'].append({
                'number': table_number,
                'dataframe': table_df,
                'description': description,
                'csv_path': csv_path,
                'html_path': html_path,
                'desc_path': desc_path
            })
            results['output_files'].extend([csv_path, html_path, desc_path])

    except Exception as e:
        _log.error(f"Error during table extraction: {e}")

    return results


# ---------------------------- IMAGE EXTRACTION ---------------------------- #
def extract_images(pdf_bytes: bytes, pdf_name: str) -> dict:
    results = {
        'images': [],
        'output_files': []
    }

    image_dir = Path("extracted_images")
    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image = Image.open(io.BytesIO(image_bytes))
                image_path = image_dir / f"{pdf_name}_page{page_index+1}_img{img_index+1}.{image_ext}"
                image.save(image_path)

                # Describe the image using local LLM
                desc = generate_image_description(image_path)
                desc_path = image_path.with_suffix(".txt")
                with open(desc_path, "w") as f:
                    f.write(desc)

                results['images'].append({
                    'page': page_index + 1,
                    'index': img_index + 1,
                    'image': image,
                    'path': image_path,
                    'description': desc,
                    'desc_path': desc_path
                })
                results['output_files'].extend([image_path, desc_path])

        doc.close()

    except Exception as e:
        _log.error(f"Error during image extraction: {e}")

    return results


# ---------------------------- MAIN ENTRY ---------------------------- #
def main():
    logging.basicConfig(level=logging.INFO)
    _log.info("Starting PDF processing...")

    try:
        file_path = input("Enter the full path to your local PDF file: ").strip()
        pdf_path = Path(file_path)

        if not pdf_path.exists():
            _log.error("The specified file path does not exist.")
            return

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

    except Exception as e:
        _log.error(f"Failed to read the PDF file: {e}")
        return

    start_time = time.time()

    _log.info("Extracting tables...")
    table_results = extract_tables(pdf_path)

    _log.info("Extracting images...")
    image_results = extract_images(pdf_bytes, pdf_path.stem)

    # Output Summary
    print("\n" + "=" * 80)
    print("PDF PROCESSING SUMMARY")
    print("=" * 80)
    print(f"File: {pdf_path.name}")
    print(f"Tables extracted: {len(table_results.get('tables', []))}")
    print(f"Images extracted: {len(image_results.get('images', []))}")
    print("-" * 80)

    for table in table_results.get('tables', []):
        print(f"\nTABLE {table['number']}:")
        print(table['dataframe'].head(3).to_markdown(index=False))
        print("\nDESCRIPTION:")
        print(table['description'])
        print("-" * 40)

    if image_results.get('images'):
        print("\nEXTRACTED IMAGES:")
        plt.figure(figsize=(15, 10))
        for i, img_data in enumerate(image_results['images'][:4]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(img_data['image'])
            plt.title(f"Page {img_data['page']} Image {img_data['index']}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        for img in image_results['images']:
            print(f"\nPAGE {img['page']} IMAGE {img['index']} DESCRIPTION:")
            print(img['description'])
            print("-" * 40)

    all_files = table_results.get('output_files', []) + image_results.get('output_files', [])
    if all_files:
        print("\nThe following files were generated and saved locally:")
        for fpath in all_files:
            print(f" - {fpath.resolve()}")

    _log.info(f"Process completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
