import csv
from datetime import datetime

def transform_csv(input_file, output_file):
    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for row in reader:
            date_str = row['Date']
            date_obj = datetime.strptime(date_str, '%m/%d/%Y %H:%M')
            formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
            
            writer.writerow({
                'Date': formatted_date,
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Volume': 0  # Assuming Volume is always 0 as per the example
            })

if __name__ == "__main__":
    input_file = 'data/input.csv'
    output_file = 'data/formatted_data.csv'
    transform_csv(input_file, output_file)