#flair model for extracting the ground truth
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter
from tqdm import tqdm

#Loading the NER model
tagger = SequenceTagger.load("flair/ner-english")

#Reading the input file
with open("klein_bestand_2.txt", "r", encoding="utf-8") as f:
    text = f.read()

#Split the text into sentences
splitter = SegtokSentenceSplitter()
sentences = splitter.split(text)

#Collect entities as list of dicts for DataFrame
entity_list = []

for sent_batch in tqdm([sentences[i:i+32] for i in range(0, len(sentences), 32)], desc="Flair NER (batch)"):
    tagger.predict(sent_batch)
    for sentence in sent_batch:
        for entity in sentence.get_spans('ner'):
            label = entity.get_label("ner").value
            if label in ("PER", "PERSON", "ORG", "ORGANIZATION", "LOC", "LOCATION", "GPE", "DATE"):
                # Standardize label
                if label in ("PER", "PERSON"):
                    norm_label = "PERSON"
                elif label in ("ORG", "ORGANIZATION"):
                    norm_label = "ORGANIZATION"
                elif label in ("LOC", "LOCATION", "GPE"):
                    norm_label = "LOCATION"
                elif label == "DATE":
                    norm_label = "DATE"
                else:
                    norm_label = label
                entity_list.append({
                    "Label": norm_label,
                    "Entity": entity.text.strip()
                })

#Save to CSV
df = pd.DataFrame(entity_list)
df.drop_duplicates(inplace=True)
df.to_csv("flair_entities_new.csv", index=False)

print("Entities saved to flair_entities.csv")
