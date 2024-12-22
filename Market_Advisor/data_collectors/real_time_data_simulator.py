import pandas as pd
import time
import csv
import os

input_file = "data/input.csv"
output_file = "data/real_time_data.csv"
data = pd.read_csv(input_file)
last_written_row = 0

if os.path.exists(output_file):
    existing_data = pd.read_csv(output_file)
else:
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data.columns)  

for index, row in data.iloc[last_written_row:].iterrows():
    with open(output_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"Written row {index + 1}: {row.to_dict()}")
    time.sleep(2.5)