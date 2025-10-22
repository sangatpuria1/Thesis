import spacy
from collections import defaultdict
import csv
import re
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def main():
    #Defining the NLP model with a maximum length of 100000 per batch
    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 100000

    #Cleans the text without punctuations and other unnecessary words/characters
    def clean_text(text):
        text = re.sub(r"\b(?:http|www)\S+\b", "", text)
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
        return re.sub(r"\s+", " ", text).strip()

    #Process text in chunks with progress tracking
    def process_with_progress(text, chunk_size=100000):
        entities = defaultdict(list)
        chunks = []
        start = 0

        #Prepare chunks
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > 0:
                    end = start + last_space
                    chunk = text[start:end]
            chunks.append(chunk)
            start = end + 1 if end < len(text) else end

        #Processing chunks
        print(f"Processing {len(chunks)} chunks...")
        for chunk in tqdm(chunks, desc="Batches", unit="chunk"):
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]:
                    entities[ent.label_].append(ent.text)

            tqdm.write(f"Batch done - Current counts: "
                       f"{', '.join([f'{k}: {len(v)}' for k, v in entities.items()])}")

        return entities

    #Generating Wordcloud for each entity separately
    def generate_wordclouds(entities):
        for entity_type, values in entities.items():
            if not values:
                continue

            #Create frequency dictionary
            freq = defaultdict(int)
            for value in values:
                freq[value] += 1

            #Generate word cloud
            wc = WordCloud(
                width=1200,
                height=800,
                background_color='white',
                colormap='viridis',
                max_words=50
            ).generate_from_frequencies(freq)

            #Display and save
            plt.figure(figsize=(12, 8))
            plt.imshow(wc, interpolation='bilinear')
            plt.title(f"Word Cloud for {entity_type} Entities", fontsize=20)
            plt.axis('off')
            plt.tight_layout()

            #Save to file
            filename = f"wordcloud_{entity_type}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {filename}")

    #Read input file
    print("📖 Reading input file...")
    try:
        with open("combined_content.txt", "r", encoding="utf-8") as f:
            text = clean_text(f.read())
        print(f"Text length: {len(text):,} characters")
    except FileNotFoundError:
        print("Error: combined_content.txt not found")
        return

    #Process text
    entities = process_with_progress(text)

    #Save raw entities
    output_file = "entities_with_duplicates.csv"
    print(f"\nSaving all entities to {output_file}...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Entity Type", "Value"])
        for entity_type, values in entities.items():
            writer.writerows([[entity_type, value] for value in values])

    #Generate word clouds
    print("\nGenerating word clouds...")
    generate_wordclouds(entities)

    print(f"\nProcessed {sum(len(v) for v in entities.values()):,} total entities")
    print("Word clouds saved as: wordcloud_PERSON.png, wordcloud_ORG.png, etc.")


if __name__ == "__main__":
    main()