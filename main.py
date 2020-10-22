
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
import ast
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import os
import pickle
from statistics import mean
import numpy as np
import time

from params import Params

class Main(object):

    def __init__(self):
        pass

    @staticmethod
    def load_dataset(data_dir):
        sentence_list = []
        label_list = []
        with open(data_dir, encoding="utf-8") as f:
            lines = f.readlines()

        sentence = ""
        label = ""
        for line in lines:
            line = line.rstrip().lower() # do lower
            if line != "":
                # print(line)
                elements = line.split("	_	_	_	")
                sentence += elements[0].split("	")[1].rstrip() + " "  # get word
                label += ast.literal_eval(elements[-1].rstrip())["ner_tag"] + " "  # get label
            elif line == "":
                # print("--- ", line)
                sentence_list.append(sentence.rstrip())
                label_list.append(label.rstrip())
                sentence = ""
                label = ""
        return pd.DataFrame(zip(sentence_list, label_list), columns=["text", "label"])

    @staticmethod
    def do_lowercase(data):
        data["text"] = data["text"].map(lambda x: str(x).lower())
        data["label"] = data["label"].map(lambda x: str(x).lower())
        return data

    @staticmethod
    def do_preprocessing(data):
        data = Main.do_lowercase(data)
        return data

    @staticmethod
    def find_max_mean_min_sentence_size(df):
        df["text_size"] = df["text"].apply(lambda x: len(str(x).split()))
        max_sentence_size = df["text_size"].max()
        mean_sentence_size = int(df["text_size"].mean())
        min_sentence_size = df["text_size"].min()
        Params.max_sentence_size = 100  # mean_sentence_size
        print("max_sentence_size (on X_train): ", max_sentence_size)
        print("mean_sentence_size (on X_train): ", mean_sentence_size)
        print("min_sentence_size (on X_train): ", min_sentence_size)
        # exit()

    @staticmethod
    def find_unique_labels(train_df, test_df):
        tag2id = {}
        tag_values = []
        idx = 0
        df_tags = pd.concat([train_df["label"], test_df["label"]])
        for tags in df_tags: # tags: str
            for tag in tags.split():
                if tag not in tag2id:
                    tag2id[tag] = idx
                    idx += 1
        tag2id[Params.padding_tag] = idx
        for key in tag2id:
            tag_values.append(key) # metric hesaplarken kullanılıyor
        Params.tag2id = tag2id
        Params.tag_values = tag_values
        print(Params.tag2id)
        print(Params.tag_values)

    @staticmethod
    def tokenize_and_preserve_labels(sentence, text_labels):
        tokenized_sentence = []
        labels = []

        # print(sentence)
        # print(type(sentence))
        # print(text_labels)

        for (word, label) in zip(sentence, text_labels):
            # print(word)
            # print(label)
            # exit()

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = Params.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            # print("tokenized_word: ", tokenized_word)
            # print("n_subwords: ", n_subwords)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    @staticmethod
    def create_dataloader(df, mode="train"):
        df["text"] = df["text"].apply(lambda x: str(x).split())
        df["label"] = df["label"].apply(lambda x: str(x).split())

        # hızlı çalışsın diye subsample alma kodu !!!
        if Params.train_sample_size != None and Params.test_sample_size != None:
            if mode == "train":
                sentences, labels = df["text"][:Params.train_sample_size], df["label"][:Params.train_sample_size]
            elif mode == "test":
                sentences, labels = df["text"][:Params.test_sample_size], df["label"][:Params.test_sample_size]
        else:
            sentences, labels = df["text"], df["label"]

        tokenized_texts_and_labels = [Main.tokenize_and_preserve_labels(sent, labs)
                                      for sent, labs in zip(sentences, labels)]
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

        input_ids = pad_sequences([Params.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=Params.MAX_LEN, dtype="long", value=0.0,
                                  truncating="post", padding="post")

        tags = pad_sequences([[Params.tag2id.get(l) for l in lab] for lab in labels],
                             maxlen=Params.MAX_LEN, value=Params.tag2id[Params.padding_tag], padding="post",
                             dtype="long", truncating="post")

        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

        inputs_tensor = torch.tensor(input_ids)
        tags_tensor = torch.tensor(tags)
        masks_tensor = torch.tensor(attention_masks)

        data = TensorDataset(inputs_tensor, masks_tensor, tags_tensor)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=Params.bs)

        return dataloader

    @staticmethod
    def get_dataloaders(train_df, test_df):
        # preprocess
        train_df = Main.do_preprocessing(train_df)
        test_df = Main.do_preprocessing(test_df)

        # find max/mean/min sentence size on train data
        Main.find_max_mean_min_sentence_size(train_df)

        # find unique labels
        Main.find_unique_labels(train_df, test_df)

        # create dataloaders
        train_dataloader = Main.create_dataloader(train_df)
        test_dataloader = Main.create_dataloader(test_df)

        return train_dataloader, test_dataloader

    @staticmethod
    def plot_loss(train_loss, test_loss):
        # loss figure
        plt.clf()
        plt.plot(train_loss, label='train')
        plt.plot(test_loss, label='valid')
        plt.title('train-valid loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        # plt.show()
        plt.savefig(os.path.join(Params.plot_dir, "loss.png"))

    @staticmethod
    def plot_f1(train_f1, test_f1, mode="micro"):
        # f1 figure
        plt.clf()
        plt.plot(train_f1, label='train')
        plt.plot(test_f1, label='valid')
        plt.title('train-valid f1-'+mode)
        plt.ylabel('f1-'+mode)
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(Params.plot_dir, "f1-"+mode+".png"))

    @staticmethod
    def create_optimizer():
        if Params.FULL_FINETUNING:
            param_optimizer = list(Params.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(Params.model.classifier.named_parameters()) # "classifier": last layer name
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        Params.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=Params.AdamW_lr,
            eps=Params.AdamW_eps,
            weight_decay=Params.weight_decay
        )

    @staticmethod
    def create_scheduler(train_dataloader):
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * Params.epoch

        # Create the learning rate scheduler.
        Params.scheduler = get_linear_schedule_with_warmup(
            Params.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

    @staticmethod
    def run_train_test(train_dataloader, test_dataloader):
        # create model
        # create from model name, uses default config
        Params.model = BertForTokenClassification.from_pretrained(
            Params.bert_model_name,
            num_labels=len(Params.tag2id),
            output_attentions=False,
            output_hidden_states=False
        )

        print("---model parameters---")
        for name, module in Params.model.named_modules():
            print(name)
            print(module)
            print()

        # push model to gpu
        if Params.use_cuda:
            Params.model.cuda()
        model_total_params = sum(p.numel() for p in Params.model.parameters())
        print("model_total_params: ", model_total_params)

        # create optimizer
        Main.create_optimizer()

        # create schedular
        Main.create_scheduler(train_dataloader)

        train_loss = []
        train_f1_micro = []
        train_f1_macro = []
        train_f1_weighted = []
        train_elapsed_time = []

        test_loss = []
        test_f1_micro = []
        test_f1_macro = []
        test_f1_weighted = []
        test_elapsed_time = []

        for epoch in range(1, Params.epoch + 1):
            print(epoch, " .epoch başladı ...")
            # train
            train_start = time.time()
            _train_loss = Main.run_train(train_dataloader)
            train_elapsed_time.append(str(time.time() - train_start))
            train_loss.append(_train_loss)

            # test
            test_start = time.time()
            _test_loss, _test_f1_micro, _test_f1_macro, _test_f1_weighted = Main.run_test(test_dataloader)
            test_elapsed_time.append(str(time.time() - test_start))
            test_loss.append(_test_loss)
            test_f1_micro.append(_test_f1_micro)
            test_f1_macro.append(_test_f1_macro)
            test_f1_weighted.append(_test_f1_weighted)

            # info
            print("train loss -> ", _train_loss)

            print("test loss -> ", _test_loss)
            print("test f1-micro -> ", _test_f1_micro)
            print("test f1-macro -> ", _test_f1_macro)
            print("test f1-weighted -> ", _test_f1_weighted)

        # save tokenizer
        Params.tokenizer.save_pretrained(Params.model_dir)
        # save model
        Params.model.save_pretrained(Params.model_dir)
        # save tag2id dict as pickle
        with open(os.path.join(Params.model_dir, Params.tag2id_pickle_name), 'wb') as handle:
            pickle.dump(Params.tag2id, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # plot
        Main.plot_loss(train_loss, test_loss)
        Main.plot_f1([1 for i in range(len(test_f1_micro))], test_f1_micro, mode="micro") # train_f1_micro
        Main.plot_f1([1 for i in range(len(test_f1_macro))], test_f1_macro, mode="macro")
        Main.plot_f1([1 for i in range(len(test_f1_weighted))], test_f1_weighted, mode="weighted")

        # save txt (delete later)
        f = open(os.path.join(Params.plot_dir, "f1-scores.txt"), "w")
        f.write("test elapsed time\n")
        f.write(str(test_elapsed_time) + "\n")
        f.write("test f1-micro\n")
        f.write(str(test_f1_micro) + "\n")
        f.write("test f1-macro\n")
        f.write(str(test_f1_macro) + "\n")
        f.write("test f1-weighted\n")
        f.write(str(test_f1_weighted) + "\n")
        f.close()

    @staticmethod
    def calculate_metrics(pred_tags, y_tags, mode="train"):
        pred_tags = list(map(lambda x: Params.tag2id[x], pred_tags))  # convert numeric type
        #print("pred_tags: ", pred_tags)
        print(list(set(pred_tags)))
        print(len(pred_tags))
        y_tags = list(map(lambda x: Params.tag2id[x], y_tags))  # convert numeric type
        #print("y_tags: ", y_tags)
        print(list(set(y_tags)))
        print(len(y_tags))
        print(Params.tag_values)

        print("---classification_report--- " + mode)
        print(classification_report(y_tags, pred_tags, target_names=Params.tag_values[:-1])) # -1: ignore _padding_ tag :)
        f1_micro = f1_score(y_tags, pred_tags, average='micro')
        f1_macro = f1_score(y_tags, pred_tags, average='macro')
        f1_weighted = f1_score(y_tags, pred_tags, average='weighted')
        return f1_micro, f1_macro, f1_weighted

    @staticmethod
    def run_train(dataloader):
        Params.model.train()  # set train mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, batch in enumerate(dataloader): # batch
            print("train batch id: ", id)

            # add batch to gpu
            if torch.cuda.is_available():
                batch = tuple(t.to(Params.device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            b_input_ids = b_input_ids.type(torch.LongTensor)
            b_input_mask = b_input_mask.type(torch.LongTensor)
            b_labels = b_labels.type(torch.LongTensor)

            if torch.cuda.is_available():
                b_input_ids = b_input_ids.to(Params.device)
                b_input_mask = b_input_mask.to(Params.device)
                b_labels = b_labels.to(Params.device)

            Params.model.zero_grad()
            outputs = Params.model(b_input_ids, token_type_ids=None,
                                   attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            epoch_loss.append(loss.item())  # save batch loss

            Params.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=Params.model.parameters(), max_norm=Params.max_grad_norm)
            Params.optimizer.step()
            Params.scheduler.step()

            # empty_cache
            torch.cuda.empty_cache()

            # calculate metrics
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predicted_labels.extend([list(p) for p in np.argmax(logits, axis=2)])
            y_labels.extend(label_ids)

        #print(predicted_labels)
        #print(type(predicted_labels))
        #exit()

        pred_tags = [Params.tag_values[p_i] for p, l in zip(predicted_labels, y_labels)
                     for p_i, l_i in zip(p, l) if Params.tag_values[l_i] != Params.padding_tag]
        y_tags = [Params.tag_values[l_i] for l in y_labels
                  for l_i in l if Params.tag_values[l_i] != Params.padding_tag]
        #Main.calculate_metrics(pred_tags, y_tags, mode="train")
        return mean(epoch_loss)

    @staticmethod
    def run_test(dataloader):
        Params.model.eval()  # set eval mode
        epoch_loss = []
        predicted_labels = []
        y_labels = []
        for id, batch in enumerate(dataloader): # batch
            print("test batch id: ", id)

            # add batch to gpu
            if torch.cuda.is_available():
                batch = tuple(t.to(Params.device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            b_input_ids = b_input_ids.type(torch.LongTensor)
            b_input_mask = b_input_mask.type(torch.LongTensor)
            b_labels = b_labels.type(torch.LongTensor)

            if torch.cuda.is_available():
                b_input_ids = b_input_ids.to(Params.device)
                b_input_mask = b_input_mask.to(Params.device)
                b_labels = b_labels.to(Params.device)

            outputs = Params.model(b_input_ids, token_type_ids=None,
                                   attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            epoch_loss.append(loss.item())  # save batch loss

            # empty_cache
            torch.cuda.empty_cache()

            # calculate metrics
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predicted_labels.extend([list(p) for p in np.argmax(logits, axis=2)])
            y_labels.extend(label_ids)

        pred_tags = [Params.tag_values[p_i] for p, l in zip(predicted_labels, y_labels)
                     for p_i, l_i in zip(p, l) if Params.tag_values[l_i] != Params.padding_tag]
        y_tags = [Params.tag_values[l_i] for l in y_labels
                  for l_i in l if Params.tag_values[l_i] != Params.padding_tag]
        f1_micro, f1_macro, f1_weighted = Main.calculate_metrics(pred_tags, y_tags, mode="test")
        return mean(epoch_loss), f1_micro, f1_macro, f1_weighted

if __name__ == '__main__':

    print("cuda available: ", torch.cuda.is_available())

    train_df = Main.load_dataset(Params.train_dataset_dir)
    test_df = Main.load_dataset(Params.test_dataset_dir)

    train_dataloader, test_dataloader = Main.get_dataloaders(train_df, test_df)
    Main.run_train_test(train_dataloader, test_dataloader)

    """
    for id, batch in enumerate(test_dataloader):
        print(id)
        b_input_ids, b_input_mask, b_labels = batch

        print(b_input_ids)
        print(b_input_ids.size())
        print(b_input_mask)
        print(b_input_mask.size())
        print(b_labels)
        print(b_labels.size())
        exit()
    """

    