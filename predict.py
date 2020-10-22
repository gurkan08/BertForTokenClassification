
"""
önemli: train ederken lower() yaptıysak, predict yaparken de text lower() yapılmalı, yoksa bulamıyor :)
"""

from transformers import BertTokenizer, BertForTokenClassification
from torch.autograd import Variable
import torch
import pickle
import os
import numpy as np
import time

from params import Params

tokenizer = BertTokenizer.from_pretrained(Params.model_dir)

model = BertForTokenClassification.from_pretrained(Params.model_dir)
if Params.use_cuda:
    model.cuda()
model.eval() # eval mode
print(model)

tag2id = pickle.load(open(os.path.join(Params.model_dir, Params.tag2id_pickle_name), "rb"))
print(tag2id)
id2tag = dict((v, k) for k, v in tag2id.items())
print(id2tag)
tag_values = [key for key in tag2id.items()]
print(tag_values)

def _predict(X):
    # X: str
    tokenized_sentence = tokenizer.encode(X)
    if torch.cuda.is_available():
        input_ids = torch.tensor([tokenized_sentence]).cuda()
    else:
        input_ids = torch.tensor([tokenized_sentence])

    #print(input_ids)
    #exit()

    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    for token, label in zip(new_tokens, new_labels):
        print("{}\t{}".format(label, token))


if __name__ == '__main__':

    start = time.time()
    #text = "İstanbul Harbiye askeri müzesi haftaiçi hangi saatler arası açık?"
    #text = "Ankara büyükşehir belediye başkanı Mansur Yavaş kaç yaşında ?"
    text = "Gürkan Şahin 2013 yılında Yıldız Teknik Üniversitesi Bilgisayar Mühendisliğinden mezun oldu. Kısa bir süre araştırma görevlisi olarak çalıştı."
    _predict(text.lower())
    end = time.time()
    print("elapsed time: ", end-start)

