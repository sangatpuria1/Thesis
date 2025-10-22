# Import all required libraries
import os
from PIL import Image
import pytesseract
from wand.image import Image as WandImage
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

#Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def convert_wmf_emf_to_png(input_path, output_path):
    #WandImage helps with converting to ImageMagick, by loading the file as img
    with WandImage(filename=input_path) as img:
        #Convert to png and save the file as output_path to give it to the next function
        with img.convert('png') as converted:
            converted.save(filename=output_path)


def extract_text_from_image(file_path):
    try:
        #Handle WMF/EMF files
        if file_path.lower().endswith(('.wmf', '.emf')):
            temp_path = file_path + '_converted.png'
            convert_wmf_emf_to_png(file_path, temp_path)
            text = pytesseract.image_to_string(Image.open(temp_path))
            os.remove(temp_path)
        # Handle standard image files
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = pytesseract.image_to_string(Image.open(file_path))
        else:
            return None
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def extract_entities(text):
    #Extracting named entities using NLTK model
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    entities = ne_chunk(tags)
    return [(' '.join([word for word, tag in entity.leaves()]), entity.label())
            for entity in entities if isinstance(entity, Tree)]


def process_folder(folder_path):
    #Process all images in the folder
    results = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing: {filename}")
        text = extract_text_from_image(file_path)

        if text and text.strip():
            print(f"Extracted Text:\n{text}")
            entities = extract_entities(text)
            results[filename] = entities

            print("Named Entities Found:")
            for entity, label in entities:
                print(f"- {entity} ({label})")
        else:
            print("No text found in image")

    return results

#Run the processing
folder_path = "C:/Users/harpr/Documents/Master/Thesis/Data NEE"
results = process_folder(folder_path)

#Print summary
print("\n=== Final Results ===")
for filename, entities in results.items():
    print(f"\n{filename}:")
    for entity, label in entities:
        print(f"  {entity} → {label}")