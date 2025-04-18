import json
import os
import random
from Config import Config

# 假设你的 JSON 文件路径
config = Config()
input_json_path = config.input_json_path

# 读取 JSON 数据（每行一个样本）
with open(input_json_path, "r", encoding="utf-8") as f:
    samples = [json.loads(line.strip()) for line in f if line.strip()]

# 过滤掉实体占比过低的样本（比如实体占比 < 20% 的句子）
def entity_ratio(sample):
    labels = sample['labels']
    num_entity = sum(1 for l in labels if l != 'O')  # 统计实体标签的数量
    return num_entity / len(labels)

# 设置实体占比阈值，保留实体占比 > 20% 的样本
threshold = 0.2
filtered_samples = [s for s in samples if entity_ratio(s) >= threshold]

# 随机打乱
random.shuffle(filtered_samples)

# 按 8:1:1 划分
n = len(filtered_samples)
train_set = filtered_samples[:int(n * 0.8)]
dev_set = filtered_samples[int(n * 0.8):int(n * 0.9)]
test_set = filtered_samples[int(n * 0.9):]

# BIO 格式转换函数
def convert_to_bio_lines(data):
    lines = []
    for item in data:
        text = item['text']
        labels = item['labels']
        assert len(text) == len(labels), f"Mismatch: {text} vs {labels}"
        for char, label in zip(text, labels):
            lines.append(f"{char} {label}")
        lines.append("")  # 空行分隔样本
    return "\n".join(lines)

# 转换成BIO格式
bio_train = convert_to_bio_lines(train_set)
bio_dev = convert_to_bio_lines(dev_set)
bio_test = convert_to_bio_lines(test_set)

# 输出路径（和原文件同目录）
base_dir = os.path.dirname(input_json_path)
train_path = os.path.join(base_dir, "train.txt")
dev_path = os.path.join(base_dir, "dev.txt")
test_path = os.path.join(base_dir, "test.txt")

# 写入文件
with open(train_path, "w", encoding="utf-8") as f:
    f.write(bio_train)
with open(dev_path, "w", encoding="utf-8") as f:
    f.write(bio_dev)
with open(test_path, "w", encoding="utf-8") as f:
    f.write(bio_test)

train_path, dev_path, test_path
