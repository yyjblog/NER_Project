

from transformers import BertTokenizer, AlbertTokenizer
from model.bert_crf import BertCRF
from model.bilstm_crf import BiLSTM_CRF
from model.roberta_crf import RoBertaCRF




def map_model(name):
    """模型映射"""
    map_func = {
        'bilstm_crf' : BiLSTM_CRF,
        'bert_crf' : BertCRF,
        'roberta_crf' : RoBertaCRF
    }
    model = map_func.get(name, BertCRF)
    return model
    
        
        
        
def map_tokenizer(name):
    """模型映射"""
    if name == 'albert_crf':
        tokenizer = AlbertTokenizer
    else:
        tokenizer = BertTokenizer
    return tokenizer
        