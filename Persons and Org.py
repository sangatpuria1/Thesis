import spacy
import nltk
from nltk.probability import FreqDist

#Downloading punkt to split text in sentences
nltk.download('punkt')

# Load spaCy English model, I used the large version, because the small not recognized everything as it suppose to be
nlp = spacy.load("en_core_web_lg")

#Extracting entities using spacy model
def extract_entities(text, entity_type):
    doc = nlp(text)
    #Iterates and returns all named entities in doc.ents by looking at the entity label
    return [ent.text.strip() for ent in doc.ents if ent.label_ == entity_type]

#Process the file in batches of 100000 and extracts Persons and Organisations in separate variables
#Batches are created because of the memory limitations Pycharm uses
def process_file(file_path, batch_size=100000):
    persons = []
    organizations = []

    #Reads the file in chunks that are the batches above
    #Reads everything with the while loop, until the last one
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(batch_size)
            if not chunk:
                break
            persons.extend(extract_entities(chunk, "PERSON"))
            organizations.extend(extract_entities(chunk, "ORG"))

    #NLTK FreqDist is used to extract the top 10 for Persons and Organisations
    person_freq = FreqDist(persons).most_common(10)
    org_freq = FreqDist(organizations).most_common(10)

    return person_freq, org_freq

#File used to extract the data from
file_path = "klein_bestand_1.txt"
top_persons, top_organizations = process_file(file_path)

#Print the top 10 most common entities for Person and Organisation
print("Top 10 most common PERSON entities:")
for name, count in top_persons:
    print(f"{name}: {count}")

print("\nTop 10 most common ORGANIZATION entities:")
for org, count in top_organizations:
    print(f"{org}: {count}")
