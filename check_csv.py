import pandas as pd

# Load the CSV
CSV_FILE = "internship.csv"

try:
    df = pd.read_csv(CSV_FILE)

    print("\n✅ CSV Loaded Successfully!\n")

    # Print all column names
    print("🔹 Columns in CSV:")
    print(df.columns.tolist())

    print("\n🔹 First 5 Rows of CSV:")
    print(df.head().to_string(index=False))

except FileNotFoundError:
    print(f"❌ File '{CSV_FILE}' not found in this folder.")
except Exception as e:
    print(f"⚠️ Error reading CSV: {e}")
