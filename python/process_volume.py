import csv

input_file = 'python/nodl_historical_data_1 copy.csv'
output_file = 'python/nodl_historical_data_processed.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write header
    header = next(reader)
    writer.writerow(header)
    
    # Process each row
    for row in reader:
        # Convert volume from string with M to float in millions
        volume = row[5]
        if volume.endswith('M'):
            row[5] = str(float(volume[:-1]) * 1_000_000)
        elif volume.endswith('K'):
            row[5] = str(float(volume[:-1]) * 1_000)
        elif volume.endswith('W'):
            row[5] = str(float(volume[:-1]) * 10_000)
            
        writer.writerow(row)

print(f"Processed data saved to {output_file}")
