import os
import pandas as pd
import subprocess

# Set your Ollama model here
DEFAULT_MODEL = "gemma3:4b"

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

def preprocess_sheet(df):
    df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    df.columns = [str(col).strip() for col in df.columns]
    return df

def df_to_markdown(df, max_rows=10):
    return df.head(max_rows).to_markdown(index=False)

def generate_context(sheet_name, df):
    keywords = {
        "name": "It seems to contain personal or entity records.",
        "date": "This might involve events or time-series data.",
        "amount": "Likely contains financial or transaction data.",
        "invoice": "Could be related to billing or accounting.",
        "revenue": "May be a financial summary or P&L statement.",
        "cargo": "Suggests logistics or shipping data.",
        "employee": "HR or personnel information is likely."
    }

    context_clues = set()
    for col in df.columns:
        for key, message in keywords.items():
            if key in col.lower():
                context_clues.add(message)

    if not context_clues:
        context_clues.add("General tabular data found.")

    return f"### Sheet: {sheet_name}\nColumns: {', '.join(df.columns)}\n" + " ".join(context_clues)

def call_ollama_gemma(prompt, model=DEFAULT_MODEL):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")

def write_summary_to_file(output_dir, base_file_name, sheet_name, summary):
    safe_sheet = "".join(c if c.isalnum() or c in " _-" else "_" for c in sheet_name)
    safe_base = os.path.splitext(os.path.basename(base_file_name))[0]
    file_name = f"{safe_base}_{safe_sheet}_summary.txt"
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✅ Summary written to: {output_path}")

def process_single_file(file_path, model=DEFAULT_MODEL):
    try:
        sheets = read_file(file_path)
    except Exception as e:
        print(f"❌ Skipping file '{file_path}': {e}")
        return

    for sheet_name, df in sheets.items():
        try:
            print(f"\n📄 Processing Sheet: {sheet_name} in '{os.path.basename(file_path)}'")
            cleaned_df = preprocess_sheet(df)

            if cleaned_df.empty:
                print(f"⚠️ Skipped - Empty after cleaning.")
                continue

            markdown = df_to_markdown(cleaned_df)
            context = generate_context(sheet_name, cleaned_df)

            prompt = (
                f"{context}\n\n"
                f"{markdown}\n\n"
                "You are an expert data analyst tasked with interpreting the contents of a spreadsheet sheet. "
                "Carefully analyze the following tabular data and answer the following:\n\n"
                "1. Provide a thorough description of what this sheet represents.\n"
                "2. Identify and explain the types of data present based on column headers and values.\n"
                "3. Suggest possible real-world use cases for this kind of data.\n"
                "4. Highlight any noticeable trends, correlations, anomalies, or patterns in the dataset.\n"
                "5. Determine whether the data looks complete and consistent or if there are signs of missing, noisy, or malformed data.\n"
                "6. Infer the business domain this sheet might belong to (e.g., finance, HR, logistics, healthcare, sales, etc.).\n"
                "Respond in a professional and structured format. Use bullet points or numbered lists for clarity wherever appropriate. Be concise but highly informative."
                "Don't ask back any questions. Just provide the best description possible"
            )

            print(f"🧠 Querying model '{model}'...")
            response = call_ollama_gemma(prompt, model=model)
            write_summary_to_file(os.path.dirname(file_path), file_path, sheet_name, response)

        except Exception as err:
            print(f"❌ Error while processing sheet '{sheet_name}': {err}")

def process_all_files_in_folder(folder_path, model=DEFAULT_MODEL):
    supported_ext = [".xlsx", ".xls", ".csv"]
    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in supported_ext
    ]

    if not all_files:
        print("❌ No valid Excel or CSV files found in the folder.")
        return

    print(f"📂 Found {len(all_files)} files to process...\n")

    for file_path in all_files:
        print(f"\n==============================")
        print(f"📑 Processing File: {os.path.basename(file_path)}")
        print(f"==============================")
        process_single_file(file_path, model)

# Entry point
if __name__ == "__main__":
    folder_path = input("Enter the full path to the folder containing Excel/CSV files: ").strip()
    if not os.path.isdir(folder_path):
        print("❌ Error: Directory not found.")
    else:
        process_all_files_in_folder(folder_path)
        
# pip install pandas openpyxl tabulate xlrd
# https://g.co/gemini/share/a692daf238d3
