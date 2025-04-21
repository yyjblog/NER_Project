from collections import Counter

from Config import Config
from process.Predictor import Predictor

# from Config import Config
# label_counter = Counter()
# config = Config()
# with open(config.path_dataset + "train.txt", "r", encoding="utf-8") as f:
#     for line in f:
#         if line.strip():
#             _, label = line.strip().split()
#             label_counter[label] += 1
# print(label_counter)
config = Config()
predictor = Predictor(config,test_loader = None)
"""
return {
    'tokens': cleaned_tokens,
    'labels': cleaned_labels,
    'entities': bio_entities
}
"""
res = predictor.predict_text("2009年11月至2012年5月，任浙江明牌珠宝股份有限公司董事会秘书；")
# 检查标签对齐

print("🟢 Tokens:")
print(res["tokens"])

print("\n🔵 Labels:")
print(res["labels"])

print("\n🟣 Entities:")
for entity in res["entities"]:
    print(entity)
