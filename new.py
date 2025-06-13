# Import necessary libraries
import logging
import time
from pathlib import Path
import pandas as pd
from docling.document_converter import DocumentConverter
import fitz  # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import os
import ollama  # Local LLM interface

# Set up logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# === Table Description Generator ===
def generate_table_description(table_df: pd.DataFrame) -> str:
    _log.info("Generating description using local Ollama model (table)...")
    table_markdown = table_df.to_markdown(index=False)

    prompt = f"""
You are a professional data analyst. Your job is to write a clear, concise, and insightful summary of the table provided. Your explanation should be easy for a general audience to understand, while still offering a complete understanding of the table's purpose and key details. Follow these instructions carefully:

1. Explain the Table’s Purpose.
2. Highlight Key Data Points or Trends.
3. Interpret Financial/Numerical Data Clearly.
4. Describe with Full Context, Not Just Numbers.
5. Keep it Short and Neat (1 paragraph).

Here is the table data:
---
{table_markdown}
---
"""

    try:
        response = ollama.chat(model="gemma:3b", messages=[
            {"role": "user", "content": prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        _log.error(f"Error generating table description: {e}")
        return "Error: Could not generate a description for this table."


# === Image Description Generator ===
def generate_image_description(image_path: Path) -> str:
    _log.info(f"Generating description using local Ollama model (image): {image_path.name}")

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = """
You are an expert analyst. Please describe the content of this image in detail. 
Explain what kind of chart, graph, figure, or scene it is, and what it conveys or visualizes.
"""

        response = ollama.chat(
            model="gemma:3b",
            messages=[{"role": "user", "content": prompt}],
            images=[image_bytes]
        )
        return response['message']['content'].strip()

    except Exception as e:
        _log.error(f"Error generating image description: {e}")
        return "Error: Could not generate description for this image."


# === Table Extraction ===
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
        conv_res = doc_converter.convert(str(pdf_path))

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


# === Image Extraction ===
def extract_images(pdf_path: Path) -> dict:
    results = {
        'images': [],
        'output_files': []
    }

    image_dir = Path("extracted_images")
    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(str(pdf_path))
        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image = Image.open(io.BytesIO(image_bytes))
                save_path = image_dir / f"{pdf_path.stem}_page{page_index+1}_img{img_index+1}.{image_ext}"
                image.save(save_path)

                # Generate description for the image
                desc_path = image_dir / f"{pdf_path.stem}_page{page_index+1}_img{img_index+1}-description.txt"
                description = generate_image_description(save_path)
                with desc_path.open("w", encoding="utf-8") as f:
                    f.write(description)

                results['images'].append({
                    'page': page_index + 1,
                    'index': img_index + 1,
                    'image': image,
                    'path': save_path,
                    'description': description,
                    'desc_path': desc_path
                })
                results['output_files'].extend([save_path, desc_path])

        doc.close()

    except Exception as e:
        _log.error(f"Error during image extraction: {e}")

    return results


# === Main ===
def main():
    _log.info("==== PDF Analysis with Local Ollama (Gemma 3) ====")
    pdf_path_str = input("Enter the path to your PDF file: ").strip()
    pdf_path = Path(pdf_path_str)

    if not pdf_path.exists():
        _log.error(f"File not found: {pdf_path}")
        return

    start_time = time.time()

    # Tables
    _log.info("Extracting tables...")
    table_results = extract_tables(pdf_path)

    # Images
    _log.info("Extracting images...")
    image_results = extract_images(pdf_path)

    # Summary
    print("\n" + "="*80)
    print("PDF PROCESSING SUMMARY")
    print("="*80)
    print(f"File: {pdf_path.name}")
    print(f"Tables extracted: {len(table_results.get('tables', []))}")
    print(f"Images extracted: {len(image_results.get('images', []))}")
    print("-"*80)

    # Table outputs
    for table in table_results.get('tables', []):
        print(f"\nTABLE {table['number']}:")
        print(table['dataframe'].head(3).to_markdown(index=False))
        print("\nDESCRIPTION:")
        print(table['description'])
        print("-"*40)

    # Image descriptions
    for img_data in image_results.get('images', []):
        print(f"\nIMAGE — Page {img_data['page']} Index {img_data['index']}")
        print(f"File: {img_data['path'].name}")
        print("DESCRIPTION:")
        print(img_data['description'])
        print("-"*40)

    # Preview thumbnails (optional)
    if image_results['images']:
        try:
            print("\nIMAGE THUMBNAILS (first 4 only):")
            plt.figure(figsize=(15, 10))
            for i, img_data in enumerate(image_results['images'][:4]):
                plt.subplot(2, 2, i+1)
                plt.imshow(img_data['image'])
                plt.title(f"Page {img_data['page']} Img {img_data['index']}")
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            _log.warning(f"Image preview skipped: {e}")

    _log.info(f"Process completed in {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()
