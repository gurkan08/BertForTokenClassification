
from transformers import BertTokenizer
import torch
import os

class Params(object):

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    plot_dir = "plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    train_dataset_dir = "data/turkish-ner-train.conllu"
    test_dataset_dir = "data/turkish-ner-test.conllu"
    tag2id = {}
    tag2id_pickle_name = "tag2id.pickle"
    max_sentence_len = 64 # None
    batch_size = 32
    batch_shuffle = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    optimizer = None
    epoch = 5
    scheduler = None
    AdamW_lr = 3e-5
    AdamW_eps = 1e-8
    weight_decay = 0.0 # default
    max_grad_norm = 1.0

    # new
    FULL_FINETUNING = True
    padding_tag = "_padding_" # "o"
    tag_values = None
    MAX_LEN = 75
    bs = 64 #64
    train_sample_size = 10000 #None
    test_sample_size = 5000 #None
    #train_test_split_seed = 39
    #test_split_size = 0.3
    bert_model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
