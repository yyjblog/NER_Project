from collections import Counter

from Config import Config
from process.Predictor import Predictor

from Config import Config
label_counter = Counter()
config = Config()
with open(config.path_dataset + "train.txt", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            _, label = line.strip().split()
            label_counter[label] += 1
print(label_counter)
#
# ALLOWED_DATASETS = ['CNER', 'CLUENER', 'CMEEE']
# ALLOWED_MODELS = ['bert_crf', 'bilstm_crf', 'roberta_crf']
# config = Config()
# config.dataset = "CMEEE"
# config.model_name = "roberta_crf"
# config.update_paths()
# predictor = Predictor(config,test_loader = None)
# """
# return {
#     'tokens': cleaned_tokens,
#     'labels': cleaned_labels,
#     'entities': bio_entities
# }
# """
# res = predictor.predict_text("大量给予肾上腺皮质激素还可能加重机体的应激状态，也会造成严重的继发性感染使病情加重。")
# # 检查标签对齐
#
# print("🟢 Tokens:")
# print(res["tokens"])
#
# print("\n🔵 Labels:")
# print(res["labels"])
#
# print("\n🟣 Entities:")
# for entity in res["entities"]:
#     print(entity)
