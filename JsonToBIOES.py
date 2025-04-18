import json
import os
from sklearn.model_selection import train_test_split

from Config import Config

config = Config()
input_file = config.input_json_path
# 输入和输出路径
output_dir = os.path.dirname(input_file)  # 输出目录
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, "train.txt")
dev_path = os.path.join(output_dir, "dev.txt")
test_path = os.path.join(output_dir, "test.txt")

# 加载原始数据
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f]

# BIOS → BMES 转换
def bios_to_bmes(labels):
    bmes_labels = []
    for i in range(len(labels)):
        curr = labels[i]
        prev = labels[i - 1] if i > 0 else 'O'
        next_ = labels[i + 1] if i < len(labels) - 1 else 'O'

        if curr.startswith('B-'):
            if next_.startswith('I-') and next_[2:] == curr[2:]:
                bmes_labels.append('B-' + curr[2:])
            else:
                bmes_labels.append('S-' + curr[2:])
        elif curr.startswith('I-'):
            if next_.startswith('I-') and next_[2:] == curr[2:]:
                bmes_labels.append('M-' + curr[2:])
            else:
                bmes_labels.append('E-' + curr[2:])
        else:
            bmes_labels.append(curr)  # O 不变
    return bmes_labels

# 构建 BIOES 格式文本
def convert_to_bioes_lines(data):
    lines = []
    for item in data:
        text = item["text"]
        labels = bios_to_bmes(item["labels"])
        assert len(text) == len(labels), f"长度不匹配: {text} vs {labels}"
        for char, label in zip(text, labels):
            lines.append(f"{char} {label}")
        lines.append("")  # 用空行分隔句子
    return "\n".join(lines)

# 划分数据集
train_set, tmp = train_test_split(data, test_size=0.2, random_state=42)
dev_set, test_set = train_test_split(tmp, test_size=0.5, random_state=42)

# 转换格式并写入文件
with open(train_path, "w", encoding="utf-8") as f:
    f.write(convert_to_bioes_lines(train_set))

with open(dev_path, "w", encoding="utf-8") as f:
    f.write(convert_to_bioes_lines(dev_set))

with open(test_path, "w", encoding="utf-8") as f:
    f.write(convert_to_bioes_lines(test_set))

print("转换完成：BMES 格式数据已生成。")
