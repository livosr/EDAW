

def count_entities(data):
    entity_count = {}
    current_entity = None

    for line in data.split('\n'):
        line = line.strip()
        if line:
            word, tag = line.split()
            if tag.startswith('B-'):
                if current_entity:
                    entity_count[current_entity] = entity_count.get(current_entity, 0) + 1
                current_entity = tag[2:]
            elif tag.startswith('I-'):
                continue
            else:
                if current_entity:
                    entity_count[current_entity] = entity_count.get(current_entity, 0) + 1
                current_entity = None

    if current_entity:
        entity_count[current_entity] = entity_count.get(current_entity, 0) + 1

    return entity_count

file_path = "train.txt"

with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

entity_counts = count_entities(data)
print(len(entity_counts))
for entity, count in entity_counts.items():
    print(f"{entity}: {count}")

print(entity_counts.keys())
ent_list = list(entity_counts.keys())
ent_list = sorted(ent_list, key=lambda x: x[0])
print(ent_list)