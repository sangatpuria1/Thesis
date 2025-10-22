import spacy
from collections import Counter

#Loading the English model with NEE
nlp = spacy.load("en_core_web_sm")


#Extracts 4-digit years from named entities labeled as DATE.
def extract_years(text):
    years = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DATE":
            for token in ent:
                if token.text.isdigit() and len(token.text) == 4:
                    years.append(token.text)
    return years

#Read the file in batches, extracts the years, counts and returns the 10 most common years
def process_file(file_path, batch_size=100000):
    year_counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(batch_size)
            if not chunk:
                break
            years = extract_years(chunk)
            year_counter.update(years)
    return year_counter.most_common(10)

#The file where the dates are extracted from
file_path = "combined_content.txt"
top_10_years = process_file(file_path)
print("Top 10 most common years found:", top_10_years)
