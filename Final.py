#This Python script compares the 4 NER-models and delivers the three circle diagrams
#Flair data and the data from klein_bestand is already created in Flair.py and Split.py

#Library imports
import os
import pandas as pd
from tqdm import tqdm
import tempfile
import spacy
from gliner import GLiNER
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import matplotlib.pyplot as plt
import re

#Hugging Face cache setup
#Set local cache directories for Hugging Face to avoid repeated downloads
cache_dir = os.path.join(tempfile.gettempdir(), "hf_cache")
for key, val in [
    ("HF_HOME", cache_dir),
    ("HF_HUB_CACHE", os.path.join(cache_dir, "hub")),
    ("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets")),
    ("HF_TRANSFORMERS_CACHE", os.path.join(cache_dir, "models")),
    ("TRANSFORMERS_CACHE", os.path.join(cache_dir, "models"))
]:
    os.environ[key] = val

#Read the import files
#Open and read the raw text data from a file
with open("klein_bestand_2.txt", "r", encoding="utf-8") as f:
    text = f.read()
#Load the ground truth (named gold_df) that is created with flair
gold_df = pd.read_csv("flair_entities_new.csv")

#Label mapping and text validation
#Function to check if a string is non-empty and valid, it is crucial for data quality and consistency
def valid_text(x):
    return isinstance(x, str) and x.strip() and x == x

#Function to map different model label variants
#E.g. spacy uses GPE and Flair LOC
def map_label(label):
    label = str(label).upper()
    if label in ("ORG", "ORGANIZATION"):
        return "ORGANIZATION"
    if label in ("LOC", "GPE", "LOCATION"):
        return "LOCATION"
    if label in ("PER", "PERSON"):
        return "PERSON"
    if label == "DATE":
        return "DATE"
    return None

#List of target entity types to focus on
target_entity_types = ["ORGANIZATION", "LOCATION", "PERSON", "DATE"]

#Standardize and filter gold-standard entities into a set of tuples
gold_set = set(
    (map_label(lbl), ent.strip().lower())
    for lbl, ent in zip(gold_df["Label"], gold_df["Entity"])
    if map_label(lbl) is not None and valid_text(ent)
)

#Loading the NER models
#Loading spaCy and set the max length
nlp_spacy = spacy.load("en_core_web_lg")
nlp_spacy.max_length = max(nlp_spacy.max_length, 200000)

#Load the NER-model GLiNER
model_g = GLiNER.from_pretrained("urchade/gliner_large")

#Load BERT tokenizer and model for NER, and wrap in a pipeline
bert_tok = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
bert_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
bert_nlp = pipeline("ner", model=bert_model, tokenizer=bert_tok, aggregation_strategy="simple")

#Load RoBERTa tokenizer and model for NER, and wrap in a pipeline
roberta_tok = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
roberta_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
roberta_nlp = pipeline("ner", model=roberta_model, tokenizer=roberta_tok, aggregation_strategy="simple")

#Creating chunk sizes to avoid crashes
#Function to split long text into manageable chunks (default: 100,000 characters)
def chunk_text(text, max_chars=100_000):
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]

#Function to clean entity strings (remove short/meaningless entities that are smaller than 2 characters)
def clean_entity(ent):
    """Remove obvious false positives: only digits, symbols, or length<2."""
    ent = ent.lower().strip()
    if len(ent) < 2:
        return ""
    if re.match(r"^[\d\W_]+$", ent):
        return ""
    return ent

#Function to clean and map raw (label, entity) pairs into standardized set
def mapped_entities(entities):
    #Clean and map, skip empty
    out = set()
    for lbl, ent in entities:
        mapped = map_label(lbl)
        ent_clean = clean_entity(ent)
        if mapped is not None and valid_text(ent_clean) and ent_clean:
            out.add((mapped, ent_clean))
    return out

#Create lists to store raw entities from each model
spacy_entities_raw, gliner_entities_raw, bert_entities_raw, roberta_entities_raw = [], [], [], []

#Process text in chunks, extracting entities with each model
for chunk in tqdm(chunk_text(text), desc="Processing 100k-char chunks"):
    if not chunk.strip():
        continue
    #spaCy NER
    doc = nlp_spacy(chunk)
    for ent in doc.ents:
        spacy_entities_raw.append((ent.label_, ent.text))
    #GLiNER NER
    for ent in model_g.predict_entities(chunk, labels=["PERSON", "ORG", "GPE", "DATE"]):
        ent_text = chunk[ent["start"]:ent["end"]]
        gliner_entities_raw.append((ent["label"], ent_text))
    #BERT NER
    for ent in bert_nlp(chunk):
        bert_entities_raw.append((ent["entity_group"], ent["word"]))
    #RoBERTa NER
    for ent in roberta_nlp(chunk):
        roberta_entities_raw.append((ent["entity_group"], ent["word"]))

#Clean, map, and standardize all extracted entities for each model
spacy_entities = mapped_entities(spacy_entities_raw)
gliner_entities = mapped_entities(gliner_entities_raw)
bert_entities = mapped_entities(bert_entities_raw)
roberta_entities = mapped_entities(roberta_entities_raw)

#Dictionary to collect all results by model
all_results = {
    "spaCy": spacy_entities,
    "GLiNER": gliner_entities,
    "BERT": bert_entities,
    "RoBERTa": roberta_entities
}

#Evaluates, stores and prints errors occured
#Function to compute precision, recall, F1-score, TP, FP, FN between predictions and gold
def eval_entities(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
    return precision, recall, f1, tp, fp, fn

#Function to print false positives and negatives for error analysis
def print_errors(pred, gold, entity_type, model_name, max_show=5):
    fp = pred - gold
    fn = gold - pred
    if fp or fn:
        print(f"  {model_name} ({entity_type}) Errors:")
        if fp:
            print(f"    False Positives: {list(fp)[:max_show]}")
        if fn:
            print(f"    False Negatives: {list(fn)[:max_show]}")

#List to collect evaluation metrics
metrics = []
print("\n=== Per-Entity-Type NER Model Results (Grouped) ===\n")

#For each entity type and each model, evaluate and print metrics/errors
for entity in target_entity_types:
    print(f"Entity: {entity}")
    gold_e = set([x for x in gold_set if x[0] == entity])
    for model, preds in all_results.items():
        preds_e = set([x for x in preds if x[0] == entity])
        p, r, f, tp, fp, fn = eval_entities(preds_e, gold_e)
        print(f"  {model:<10} Precision: {p:.3f} Recall: {r:.3f} F1: {f:.3f} TP: {tp} FP: {fp} FN: {fn}")
        print_errors(preds_e, gold_e, entity, model)
        metrics.append({"Model": model, "Entity": entity, "Precision": p, "Recall": r, "F1": f})
    print()

#Create the circle diagrams
#Each slice shows P, R, F1, and percentage
#Convert results to DataFrame for visualization
results_df = pd.DataFrame(metrics)

#For each entity type, make a pie chart for F1-scores per model
for entity in results_df["Entity"].unique():
    entity_data = results_df[(results_df["Entity"] == entity) & (results_df["F1"] > 0)]
    #Skip if all F1-scores are zero (no diagram for DATE)
    if entity_data.empty:
        continue

    print(f"\n=== F1-Score Distribution for {entity} ===")
    for _, row in entity_data.iterrows():
        print(f"  {row['Model']:<10}: P={row['Precision']:.3f}, R={row['Recall']:.3f}, F1={row['F1']:.3f}")

    #Label for each pie slice: Model, Precision, Recall, F1
    labels = [
        f"{row['Model']}\nP:{row['Precision']:.2f} R:{row['Recall']:.2f}\nF:{row['F1']:.2f}"
        for _, row in entity_data.iterrows()
    ]

    #Make and display the pie chart
    plt.figure()
    plt.pie(
        entity_data["F1"],
        labels=labels,
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title(f'F1-Score Distribution for {entity}')
    #Ensures the pie is a circle
    plt.axis('equal')

#Adjust layout and display all charts at once
plt.tight_layout()
plt.show()