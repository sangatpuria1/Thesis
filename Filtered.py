#Filtered
import csv
from collections import Counter


def count_top_entities(csv_file):
    counters = {
        'PERSON': Counter(),
        'ORG': Counter(),
        'GPE': Counter(),
        'DATE': Counter()
    }

    #Read CSV without header using next
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) >= 2:
                entity_type, value = row[0], row[1]
                if entity_type in counters:
                    counters[entity_type][value] += 1

    # Display top 10 for each entity type
    for entity_type, counter in counters.items():
        print(f"\nTop 10 {entity_type} entities:")
        print("-" * 40)
        for value, count in counter.most_common(10):
            print(f"{value}: {count} occurrences")
        print("-" * 40)


if __name__ == "__main__":
    csv_file = "entities_with_duplicates.csv"
    print(f"Analyzing {csv_file}...")
    count_top_entities(csv_file)