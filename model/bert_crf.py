

from torch import nn
from model.layers.crf import CRF
from transformers import BertModel, BertPreTrainedModel



class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (None, logits)
        if labels is not None:
            size = logits.size()
            labels = labels[:,:size[1]]
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,) + outputs
        return outputs # (loss), scores


# class BertCrf(BertPreTrainedModel):
#     """
#         Bert + CRF
#     """
#     def __init__(self, config):
#         super(BertCrf, self).__init__(config)
#         self.bert = BertModel(config)                                       # Bert
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)               # dropout
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # linear layer
#         self.crf = CRF(num_tags=config.num_labels, batch_first=True)        # CRF
#         self.num_labels = config.num_labels
#         self.init_weights()
#         # self.fc = nn.Linear(arg_config.bert_hidden_size, arg_config.num_labels)  

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
#         outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
#         sequence_output = outputs[0]                                        # Bert的每个token输出的logit
#         sequence_output = self.dropout(sequence_output)                     
#         logits = self.classifier(sequence_output)                           # linear layer
#         outputs = (logits,)

#         # Loss
#         # softmax
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # 截取labels长度
#             seq_len = logits.size()[1]
#             labels = labels[:,:seq_len].contiguous()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs =(loss,)+outputs

#         # # crf
#         # if labels is not None:
#         #     # logits = logits.transpose(0, 1)      # (seq_len, batch_size, num_labels)
#         #     # labels = labels.transpose(0, 1)
#         #     logits = logits[:,1:-1,:]
#         #     size = logits.size()
#         #     labels = labels[:,:size[1]]
#         #     if attention_mask is not None:
#         #         attention_mask = attention_mask[:,1:-1]#.uint8()            # 去除[CLS]、[SEP]
#         #         attention_mask = attention_mask.to(torch.uint8)
#         #         # attention_mask = attention_mask.transpose(0, 1)
#         #         loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
#         #     else:
#         #         loss = self.crf(emissions = logits, tags=labels)
#         #     outputs =(-1*loss,)+outputs

        
#         # paper loss （后续修改）
#         # pass
#         return outputs # (loss), scores
