from collections import Counter
from Config import Config
label_counter = Counter()
config = Config()
with open(config.path_dataset + "train.txt", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            _, label = line.strip().split()
            label_counter[label] += 1
print(label_counter)
