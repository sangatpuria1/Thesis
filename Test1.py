import spacy
from tqdm import tqdm


def process_large_file_with_progress(input_file, output_file, batch_size=10000):
    #Loading model
    nlp = spacy.load("en_core_web_lg")

    #Count total lines first (for accurate progress)
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    #Calculate total batches
    total_batches = (total_lines + batch_size - 1) // batch_size

    #Process with progress tracking
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        outfile.write("Entity,Type,Description\n")
        batch = []

        #Create progress bar
        pbar = tqdm(total=total_batches, desc="Processing batches", unit="batch")

        for line in infile:
            batch.append(line.strip())
            if len(batch) >= batch_size:
                #Process batch
                docs = nlp.pipe(batch)
                for doc in docs:
                    for ent in doc.ents:
                        outfile.write(f"{ent.text},{ent.label_},{spacy.explain(ent.label_)}\n")

                #Update progress
                batches_processed = pbar.n
                batches_remaining = total_batches - batches_processed - 1
                pbar.set_postfix({"remaining": batches_remaining})
                pbar.update(1)

                batch = []

        #Final batch
        if batch:
            docs = nlp.pipe(batch)
            for doc in docs:
                for ent in doc.ents:
                    outfile.write(f"{ent.text},{ent.label_},{spacy.explain(ent.label_)}\n")
            pbar.update(1)

        pbar.close()

process_large_file_with_progress("combined_content.txt", "entities5.csv")