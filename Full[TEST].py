import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

import spacy
from gliner import GLiNER
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

#Loading the ground truth
df_gold = pd.read_csv("csv_entities1.csv")
df_gold.columns = df_gold.columns.str.strip()
assert {'Label', 'Entity'}.issubset(df_gold.columns)

#Label to extract
labels_to_eval = ["ORGANIZATION", "PERSON", "LOCATION", "DATE"]

gold_dict = {}
for label in labels_to_eval:
    gold_dict[label] = set(
        e.strip().upper() for e in df_gold[df_gold['Label'].str.upper() == label]['Entity']
    )

#Loading the input file
with open("klein_bestand_2.txt", "r", encoding="utf-8") as f:
    text = f.read()

#Split the data in chunks
def split_into_chunks(text, chunk_size=100_000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

#In chunks, extracting data with spacy, gliner, BERT and RoBERTa
def extract_spacy(text):
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 10_000_000
    results = []
    for chunk in tqdm(split_into_chunks(text, 500_000), desc="spaCy chunks"):
        doc = nlp(chunk)
        for ent in doc.ents:
            results.append((ent.label_.upper(), ent.text.strip().upper()))
    return results

def extract_gliner(text):
    model = GLiNER.from_pretrained("urchade/gliner_large")
    labels = labels_to_eval
    results = []
    for chunk in tqdm(split_into_chunks(text, 100_000), desc="GLiNER chunks"):
        ents = model.predict_entities(chunk, labels)
        for ent in ents:
            results.append((ent['label'].upper(), ent['text'].strip().upper()))
    return results

def extract_transformer(text, model_name, label_map=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    results = []
    for chunk in tqdm(split_into_chunks(text, 50_000), desc=f"{model_name} chunks"):
        ner_results = nlp_pipe(chunk)
        for ent in ner_results:
            label = ent.get('entity_group', ent.get('entity')).upper()
            if label_map and label in label_map:
                label = label_map[label]
            results.append((label, ent['word'].strip().upper()))
    return results

#Label maps for BERT and RoBERTa
bert_label_map = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "GPE", "MISC": "MISC"}
roberta_label_map = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "GPE", "MISC": "MISC"}

#Run the models
print("Extracting with spaCy...")
spacy_ents = extract_spacy(text)

print("Extracting with GLiNER...")
gliner_ents = extract_gliner(text)

print("Extracting with BERT...")
bert_ents = extract_transformer(text, "dslim/bert-base-NER", bert_label_map)

print("Extracting with RoBERTa...")
roberta_ents = extract_transformer(text, "Jean-Baptiste/roberta-large-ner-english", roberta_label_map)

#Presenting the results
model_results = {
    "spaCy": spacy_ents,
    "GLiNER": gliner_ents,
    "BERT": bert_ents,
    "RoBERTa": roberta_ents
}

#Accuracy of the models
def calc_metrics(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

print("\n=== Evaluation (per label) ===")
for label in labels_to_eval:
    print(f"\nLABEL: {label}")
    gold_entities = gold_dict[label]
    for model_name, ents in model_results.items():
        pred_entities = set(e[1] for e in ents if e[0] == label)
        precision, recall, f1 = calc_metrics(pred_entities, gold_entities)
        print(f"{model_name:<10} Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
