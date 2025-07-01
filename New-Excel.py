import os
import pandas as pd
from fpdf import FPDF
import subprocess

SUPPORTED_EXT = [".xlsx", ".xls", ".csv"]
OUTPUT_FOLDER = "generated_pdfs"
DEFAULT_MODEL = "gemma3:4b"
FONT_DIR = "fonts"
UNICODE_FONT = "DejaVuSans.ttf"  # Make sure it's placed inside fonts/

# Read Excel/CSV
def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        print(f"📥 Reading Excel file: {file_path}")
        return pd.read_excel(file_path, sheet_name=None)
    elif ext == ".csv":
        print(f"📥 Reading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        return {"CSV": df}
    else:
        raise ValueError("Unsupported file type. Only .xlsx, .xls, and .csv are supported.")

# Preprocess sheet
def preprocess_sheet(df):
    df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    df.columns = [str(col).strip() for col in df.columns]
    return df

# Describe a single row using local LLM
def describe_row_with_llm(row_dict, model=DEFAULT_MODEL):
    columns = ", ".join(row_dict.keys())
    values = ", ".join([str(v) for v in row_dict.values()])

    prompt = f"""
You are a document assistant tasked with interpreting spreadsheet rows.
Given the following column headers and values, write a detailed and natural English paragraph describing the row:

Columns: {columns}
Values: {values}

Your response should be a single paragraph summarizing this row in natural language.
Do not list the fields one by one. Instead, generate a smooth descriptive narrative.
Avoid making assumptions not directly supported by the values.
"""

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=90
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.SubprocessError as e:
        return f"[ERROR] Failed to generate description: {e}"

# Generate a combined PDF with all paragraphs
def generate_narrative_pdf(descriptions, output_path, title):
    font_path = os.path.join(FONT_DIR, UNICODE_FONT)
    pdf = FPDF()
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.add_font("DejaVu", "B", font_path, uni=True)

    pdf.add_page()
    pdf.set_font("DejaVu", "B", 14)
    pdf.multi_cell(0, 10, title, align='C')
    pdf.ln(5)
    pdf.set_font("DejaVu", "", 12)

    for paragraph in descriptions:
        pdf.multi_cell(0, 10, paragraph)
        pdf.ln(5)

    pdf.output(output_path)

# Process one sheet of one file
def process_sheet_narratively(df, output_dir, base_filename, sheet_name):
    cleaned_df = preprocess_sheet(df)
    if cleaned_df.empty:
        print(f"⚠️ Skipped sheet '{sheet_name}' - Empty after cleaning.")
        return

    descriptions = []
    for idx, row in cleaned_df.iterrows():
        row_data = row.to_dict()
        print(f"🤖 Generating description for row {idx + 1}...")
        narrative = describe_row_with_llm(row_data)
        descriptions.append(narrative)

    if descriptions:
        safe_base = os.path.splitext(os.path.basename(base_filename))[0]
        safe_sheet = "".join(c if c.isalnum() or c in " _-" else "_" for c in sheet_name)
        final_filename = f"{safe_base}_{safe_sheet}_summary.pdf"
        final_path = os.path.join(output_dir, final_filename)
        title = f"{safe_base} - {sheet_name} Summary Report"
        generate_narrative_pdf(descriptions, final_path, title)
        print(f"✅ Created Summary PDF: {final_filename}")
    else:
        print(f"⚠️ No content to describe in sheet: {sheet_name}")

# Process one file (Excel/CSV)
def process_single_file(file_path):
    try:
        sheets = read_file(file_path)
    except Exception as e:
        print(f"❌ Skipping file '{file_path}': {e}")
        return

    for sheet_name, df in sheets.items():
        try:
            print(f"\n📄 Processing Sheet: {sheet_name} in '{os.path.basename(file_path)}'")
            process_sheet_narratively(df, OUTPUT_FOLDER, file_path, sheet_name)
        except Exception as err:
            print(f"❌ Error processing sheet '{sheet_name}': {err}")

# Process all supported files in a folder
def process_all_files_in_folder(folder_path):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ]

    if not all_files:
        print("❌ No valid Excel or CSV files found in the folder.")
        return

    print(f"📂 Found {len(all_files)} file(s) to process...\n")

    for file_path in all_files:
        print(f"\n==============================")
        print(f"📑 Processing File: {os.path.basename(file_path)}")
        print(f"==============================")
        process_single_file(file_path)

# Entry Point
if __name__ == "__main__":
    folder_path = input("Enter the full path to the folder containing Excel/CSV files: ").strip()
    if not os.path.isdir(folder_path):
        print("❌ Error: Directory not found.")
    else:
        process_all_files_in_folder(folder_path)
#pip install pandas openpyxl fpdf==1.7.2
