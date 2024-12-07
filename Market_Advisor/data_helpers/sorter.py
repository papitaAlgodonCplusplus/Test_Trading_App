import csv
from datetime import datetime

# Read the CSV file
with open('data/input.csv', mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)
    rows = list(reader)

# Sort the rows by the "Date" column
rows.sort(key=lambda row: datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))

# Write the sorted data back to the CSV file
with open('data/input.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
