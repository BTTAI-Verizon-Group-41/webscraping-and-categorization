import pandas as pd

# Try reading the CSV with an explicit encoding
try:
    df = pd.read_csv("data.csv", encoding="utf-8")
except UnicodeDecodeError:
    # Fallback to a different encoding if UTF-8 fails
    df = pd.read_csv("data.csv", encoding="latin1")

print(df.shape)