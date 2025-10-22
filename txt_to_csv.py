import csv

#Input and output filename
input_file = 'output_flair_1.txt'
output_file = 'csv_entities1.csv'

#Open both files above
with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    #Skip the header and the _ line for the first 2 sentences
    for _ in range(2):
        next(infile)

    #Create CSV writer
    writer = csv.writer(outfile)

    #Create the headers
    writer.writerow(['Label', 'Entity'])

    #Proces line for line
    for line in infile:
        #Split on white space and tab
        parts = line.strip().split('\t')
        if len(parts) < 2:
            parts = [p for p in line.strip().split(' ') if p]

        #Write to CSV
        if len(parts) >= 2:
            label = parts[0].strip()
            entity = ' '.join(parts[1:]).strip()
            writer.writerow([label, entity])

print(f"Conversie voltooid! Bestand opgeslagen als {output_file}")