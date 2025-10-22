import spacy
from collections import Counter
import matplotlib.pyplot as plt

#Load SpaCy model
nlp = spacy.load("en_core_web_lg")

#Load text from file
with open("klein_bestand_4.txt", "r", encoding="utf-8") as f:
    text = f.read()

#Split into chunks to avoid SpaCy memory limits
chunk_size = 500_000
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
org_names = []
print(f"Processing {len(chunks)} chunks...")

#Process each chunk
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1}/{len(chunks)}")
    doc = nlp(chunk)
    org_names.extend([
        ent.text for ent in doc.ents
        if ent.label_ == "PER" and len(ent.text.strip()) >= 4
    ])

#Count top orgs
org_counter = Counter(org_names)
top_orgs = org_counter.most_common(10)

#Unpack for plotting
labels, counts = zip(*top_orgs)

#Plot
plt.figure(figsize=(10, 6))
plt.barh(labels, counts, color="skyblue")
plt.xlabel("Frequency")
plt.title("Top 10 Most Common Organization Names (min. 4 characters)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
