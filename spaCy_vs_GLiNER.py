#GLiNER and SpaCy, and RoBERTa and BERT were separated for performance
#Later are the models combined and updated
import pandas as pd
import spacy
from gliner import GLiNER
from tqdm import tqdm
from spacy.lang.en import English

#Load Ground Truth data
df_gold = pd.read_csv("csv_entities1.csv")
df_gold.columns = df_gold.columns.str.lower()

#Extract the labels which are ORGANIZATION
gold_entities = df_gold[df_gold["label"].str.upper() == "ORGANIZATION"]
gold_entities = gold_entities.to_dict(orient="records")

#Loading the file where to extract the entities from
with open("klein_bestand_2.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

#Preparing smaller chunks of maximum 10 000
def split_into_chunks(text, chunk_size=10000):

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]

        last_period = chunk.rfind('. ')
        last_newline = chunk.rfind('\n')
        split_pos = max(last_period, last_newline)

        if split_pos > 0 and (len(chunk) - split_pos) < 500:  # Only adjust if near boundary
            chunks.append(chunk[:split_pos + 1])
            remaining = text[i + split_pos + 1:i + chunk_size]
            if remaining.strip():
                chunks.append(remaining)
        else:
            chunks.append(chunk)
    return chunks


chunks = split_into_chunks(full_text)

#Loading both models
nlp = spacy.load("en_core_web_trf")
nlp.max_length = 20_000_000  # Set high to avoid warnings
gliner = GLiNER.from_pretrained("urchade/gliner_large")
labels = ["ORGANIZATION"]


#Extracting the entities NER-model based
def extract_spacy(chunks):
    print(f"Extracting with spaCy ({len(chunks)} chunks)...")
    entities = []
    for chunk in tqdm(chunks):
        try:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_.upper() == "ORG":
                    entities.append({"text": ent.text.strip(), "label": "ORGANIZATION"})
        except Exception as e:
            print(f"Error processing spaCy chunk: {str(e)}")
            continue
    return entities


def extract_gliner(chunks):
    print(f"Extracting with GLiNER ({len(chunks)} chunks)...")
    entities = []
    for chunk in tqdm(chunks):
        try:
            result = gliner.predict_entities(chunk, labels=labels)
            entities.extend(result)
        except Exception as e:
            print(f"Error processing GLiNER chunk: {str(e)}")
            continue
    return entities


#Process in batches to avoid system crashes
batch_size = 5
spacy_pred, gliner_pred = [], []

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    spacy_pred.extend(extract_spacy(batch))
    gliner_pred.extend(extract_gliner(batch))


#Normalized entities
def normalize(entities):
    result = set()
    for e in entities:
        text = e.get("text") or e.get("entity")
        label = e.get("label")
        if text and label:
            norm = text.lower().strip().strip(".,;:()[]")
            result.add((norm, label.upper()))
    return result


gold_set = normalize(gold_entities)
pred_spacy_set = normalize(spacy_pred)
pred_gliner_set = normalize(gliner_pred)
ensemble_pred_set = pred_spacy_set.union(pred_gliner_set)


#Accuracy check
def evaluate(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


#Presenting the results
print("\n=== Evaluation Results (for ORGANIZATION entities only) ===")
print("spaCy:", evaluate(pred_spacy_set, gold_set))
print("GLiNER:", evaluate(pred_gliner_set, gold_set))
print("Ensemble:", evaluate(ensemble_pred_set, gold_set))