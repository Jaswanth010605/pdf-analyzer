!pip install PyMuPDF matplotlib docling pandas google-generativeai

# Import necessary libraries
import logging
import time
from pathlib import Path
import pandas as pd
from google.colab import files
import google.generativeai as genai
from docling.document_converter import DocumentConverter
import fitz  # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import os

# Set up logging for informative messages
_log = logging.getLogger(__name__)

# Configure Gemini API
try:
    genai.configure(api_key="Enter_Your_API_KEY")  # Replace with your actual API key
except Exception as e:
    _log.error("Failed to configure Gemini API. Please ensure you have set your API key correctly.")
    exit()

def generate_table_description(table_df: pd.DataFrame) -> str:
    """
    Generates a description for a given DataFrame using the Gemini API.
    """
    _log.info("Generating description with Gemini API...")
    table_markdown = table_df.to_markdown(index=False)

    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    prompt = f"""
    You are a professional data analyst. Your job is to write a clear, concise, and insightful summary of the table provided. Your explanation should be easy for a general audience to understand, while still offering a complete understanding of the table's purpose and key details. Follow these instructions carefully:
1. Explain the Table’s Purpose (1 sentence): Begin with a simple statement that clearly tells what the table is about (e.g., "This table shows the annual revenue of Company X from 2020 to 2024.").
2. Highlight Key Data Points or Trends: Identify and explain the most important numbers, totals, years, categories, or trends (e.g., highest/lowest values, increases or decreases over time, notable changes, outliers, etc.).
3.Interpret Financial/Numerical Data Clearly:
  Mention the units and currency (e.g., USD, INR, %).
  For time-based data (e.g., years, quarters), describe what each period shows.
  Point out any totals, averages, or subtotals and what they mean.
  If applicable, compare key values across rows or columns to show relationships or performance differences.
4.Describe with Full Context, Not Just Numbers: Instead of saying only "Revenue in 2022 was 5 million," explain: "In 2022, the company's revenue peaked at $5 million, showing a 25% increase from the previous year."
5.Keep it Short and Neat (1 paragraph): The final output should be a compact, well-structured paragraph (4–6 sentences) that gives a full and easy-to-understand picture of the table.
Avoid technical jargon. Make it informative, clear, and reader-friendly.

    Here is the table data:
    ---
    {table_markdown}
    ---
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        _log.error(f"Error communicating with Gemini API: {e}")
        return "Error: Could not generate a description for this table."

def extract_tables(pdf_path: Path) -> dict:
    """
    Extracts tables from PDF and generates descriptions
    Returns dictionary with table data and metadata
    """
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

            # Save outputs
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

def extract_images(pdf_bytes: bytes, pdf_name: str) -> dict:
    """
    Extracts images from PDF bytes
    Returns dictionary with image data and paths
    """
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
                save_path = image_dir / f"{pdf_name}_page{page_index+1}_img{img_index+1}.{image_ext}"
                image.save(save_path)

                results['images'].append({
                    'page': page_index + 1,
                    'index': img_index + 1,
                    'image': image,
                    'path': save_path
                })
                results['output_files'].append(save_path)

        doc.close()
        
    except Exception as e:
        _log.error(f"Error during image extraction: {e}")
    
    return results

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    _log.info("Starting PDF processing...")

    # Upload PDF
    _log.info("Please upload your PDF file.")
    try:
        uploaded = files.upload()
        if not uploaded:
            _log.warning("No PDF file uploaded. Exiting.")
            return
            
        file_path = list(uploaded.keys())[0]
        pdf_bytes = uploaded[file_path]
        pdf_path = Path(file_path)
        
    except Exception as e:
        _log.error(f"File upload failed: {e}")
        return

    # Process PDF
    start_time = time.time()
    
    # Extract tables
    _log.info("Extracting tables...")
    table_results = extract_tables(pdf_path)
    
    # Extract images
    _log.info("Extracting images...")
    image_results = extract_images(pdf_bytes, pdf_path.stem)
    
    # Display results
    print("\n" + "="*80)
    print("PDF PROCESSING SUMMARY")
    print("="*80)
    print(f"File: {pdf_path.name}")
    print(f"Tables extracted: {len(table_results.get('tables', []))}")
    print(f"Images extracted: {len(image_results.get('images', []))}")
    print("-"*80)
    
    # Show table summaries
    for table in table_results.get('tables', []):
        print(f"\nTABLE {table['number']}:")
        print(table['dataframe'].head(3).to_markdown(index=False))
        print("\nDESCRIPTION:")
        print(table['description'])
        print("-"*40)
    
    # Show image thumbnails
    if image_results.get('images'):
        print("\nEXTRACTED IMAGES:")
        plt.figure(figsize=(15, 10))
        for i, img_data in enumerate(image_results['images'][:4]):  # Show max 4 thumbnails
            plt.subplot(2, 2, i+1)
            plt.imshow(img_data['image'])
            plt.title(f"Page {img_data['page']} Image {img_data['index']}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Download all files
    all_files = table_results.get('output_files', []) + image_results.get('output_files', [])
    if all_files:
        _log.info("Preparing files for download...")
        for fpath in all_files:
            try:
                files.download(str(fpath))
            except Exception as e:
                _log.error(f"Failed to download {fpath.name}: {e}")
    
    _log.info(f"Process completed in {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()