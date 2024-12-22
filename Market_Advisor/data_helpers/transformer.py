import pandas as pd
from datetime import datetime

# Read the CSV file with tab separator
df = pd.read_csv('data/input.csv', sep='\t')

# Rename columns to match the desired output
df.rename(columns={'<DATE>': 'Date', '<TIME>': 'Time', '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close', '<TICKVOL>': 'Volume'}, inplace=True)

# Combine Date and Time into a single datetime column and reformat
df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Keep only the required columns
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Save to output.csv
df.to_csv('output.csv', index=False)

print("Conversion complete. The output is saved as 'output.csv'.")
