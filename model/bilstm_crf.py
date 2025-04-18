import os
from collections import Counter

import torch
from torch import nn
import pickle as pkl
from model.layers.crf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.emb_size = 300
        self.layer_num = 2
        self.hidden_size = 256
        self.num_label = len([x for x in open(config.path_tgt_map, 'r').readlines() if x.strip()])
        self.vocab_size = len(pkl.load(open(config.path_vocab, 'rb')))
        self.dropout = nn.Dropout(0.3)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, num_layers=self.layer_num,
                            bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size * 2, self.num_label)
        self.crf = CRF(num_tags=self.num_label, batch_first=True)

        # 读取标签映射
        # with open(config.path_tgt_map, 'r', encoding='utf-8') as f:
        #     label2id = {line.strip().split()[0]: int(line.strip().split()[1]) for line in f if line.strip()}
        #
        # # 直接根据 train.txt 计算 label_weights
        # train_label_path = os.path.join(config.path_dataset, 'train.txt')
        # self.label_weights = self.compute_label_weights(train_label_path, label2id).to(config.device)

    def forward(self, input_ids, labels=None, attention_mask=None):
        self.lstm.flatten_parameters()
        embeds = self.embedding(input_ids)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        output = (None, logits)

        # if labels is not None:
        #     size = logits.size()
        #     labels = labels[:, :size[1]]
        #     attention_mask = attention_mask[:, :size[1]]
        #
        #     loss_fct = nn.CrossEntropyLoss(weight=self.label_weights, reduction='none')
        #     loss = loss_fct(logits.view(-1, self.num_label), labels.view(-1))
        #     loss = loss.view(labels.size())
        #     loss = loss * attention_mask.float()
        #     loss = loss.sum(dim=1) / attention_mask.sum(dim=1)
        #     loss = loss.mean()
        #
        #     output = (loss, logits)

        if labels is not None:
            size = logits.size()
            labels = labels[:, :size[1]]
            loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask)
            output = (loss, logits)

        return output

    @staticmethod
    def compute_label_weights(label_path, label2id, min_freq=1e-3, max_weight=10000.0):
        from collections import Counter
        counter = Counter()
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        label = parts[1]
                        counter[label] += 1
        total = sum(counter.values())
        weights = []
        for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
            freq = counter.get(label, 0)
            freq = max(freq, min_freq)  # 避免除以0
            weight = total / freq
            weight = min(weight, max_weight)  # 防止极端权重
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float)
