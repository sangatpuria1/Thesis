import json

#Load your current format
with open("truth_entities.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted = []

for ent in raw_data["entities"]:
    text = ent["text"]
    label = ent["label"]
    entry = {
        "text": text,
        "entities": [
            {
                "start": 0,
                "end": len(text),
                "label": label
            }
        ]
    }
    converted.append(entry)

#Save to a new file
with open("truth_entities_converted.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2)

print("✅ Converted file saved as 'truth_entities_converted.json'")
