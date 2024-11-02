import pandas as pd

df = pd.read_csv('categorizedurls.csv', header = 0)

# Extract the ending of a URL (top-level domain)
# using pandas vectorized string operations
df['url_ending'] = df['url'].str.split('.').str[-1]

# Save the result (one column, same order) to a new csv file
df[['url_ending']].to_csv('output_with_url_endings.csv', index=False)