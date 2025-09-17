import pandas as pd

# Load the CSV file
df = pd.read_csv("labels.csv", sep="\t",dtype=str)  # Use sep="\t" if it's tab-separated

# Replace .png with .jpg in the 'frame' column
df['file'] = df['file'].str.replace('.png', '.jpg', regex=False)

# Save the updated CSV
df.to_csv("labels_jpg.csv", index=False, sep="\t")