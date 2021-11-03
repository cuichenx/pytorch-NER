from torch.utils.data import Dataset
from utils import normalize_word
import torch
import random
import numpy as np

class SCH_ElaborateExpressions(Dataset):
    def __init__(self, data_path, sent_ids, positive_ratio=0.5, switch_prob=0, is_test=False):
        self.sentences, self.tags = {}, {}
        self.l2i, self.i2l = {}, []
        self.w2i, self.i2w = {}, []
        self.c2i, self.i2c = {}, []
        self.wi2ci = {}
        # d has keys:
        # sentences: dict with 776615 keys (all sentences)
        # tags: dict with 23234 keys (only sentences with tagged EEs)
        # l2i: dict, i2l: list, length 3 (OIB)
        # w2i: dict, i2w: list, length 12939 (vocab size)
        # c2i: dict, i2c: list, length 82 (number of phonemes)
        # wi2ci: dict, length 3311 (number of valid hmong syllables)
        self.__dict__.update(torch.load(data_path))
        self.sentences = {idx: self.sentences[idx] for idx in sent_ids}  # filer only selected data
        self.tags = {idx: self.tags[idx] for idx in sent_ids if idx in self.tags}  # filer only selected data

        # positive ratio: how many positive sentences to include in the whole usable dataset
        self.positive_length = len(self.tags)

        if positive_ratio == -1:
            self.tot_length = len(self.sentences)
        else:
            assert 0 < positive_ratio <= 1.0
            self.tot_length = int(self.positive_length / positive_ratio)

        self.positive_keys = list(self.tags.keys())
        self.negative_keys = list(self.sentences.keys() - self.tags.keys())

        self.switch_prob = switch_prob  # with this probability, switch the order of the second and fourth word in an EE
        self.is_test = is_test
        if is_test:
            self.chosen_negative_keys = np.random.choice(self.negative_keys, size=self.tot_length-self.positive_length, replace=False)

        if grouped_swap_elabs is not None:  # swap these elabs in the positive sentences
            # turn list of word tuples into set of index tuples
            self.swapped_counter, self.keep_counter = 0, 0
            grouped_swap_elabs = set(tuple(self.w2i.get(w, self.w2i['UNK']) for w in ee) for ee in grouped_swap_elabs)
            for k in self.positive_keys:
                self.sentences[k] = self.switch_CC_order(self.sentences[k], self.tags[k], grouped_swap_elabs)

            print(f"swapped {len(grouped_swap_elabs)} elabs in {self.swapped_counter} sentences; keeping {self.keep_counter} sentences")


    def switch_CC_order(self, sentence, tag, grouped_swap_elabs=None):
        ''' change the first ABAC elaborate expression in sentence to ACAB '''
        begin = tag.index(self.l2i['B'])
        if begin+3 < len(sentence):
            if grouped_swap_elabs is None or tuple(sentence[begin:begin+4]) in grouped_swap_elabs:
                sentence[begin+1], sentence[begin+3] = sentence[begin+3], sentence[begin+1]
                self.swapped_counter += 1
            else:
                self.keep_counter += 1
        return sentence


    def __len__(self):
        return self.tot_length


    def __getitem__(self, idx):
        if idx >= self.positive_length:
            if self.is_test:
                sent_id = self.chosen_negative_keys[idx-self.positive_length]
            else:
                sent_id = random.choice(self.negative_keys)
        else:
            sent_id = self.positive_keys[idx]
        sentence = self.sentences[sent_id]
        tag = self.tags.get(sent_id, [self.l2i['O']] * len(sentence))
        if idx < self.positive_length and random.random() < self.switch_prob:
            sentence = self.switch_CC_order(sentence, tag)
        seq_char_list = [self.wi2ci.get(wi, [self.c2i['UNK']]*3) for wi in sentence]
        return {'text': torch.tensor(sentence),
                'label': torch.tensor(tag),
                'char': seq_char_list}


class MyDataset(Dataset):
    def __init__(self, file_path, word_vocab, label_vocab, alphabet, number_normalized):
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.alphabet = alphabet
        self.number_normalized = number_normalized
        texts, labels = [], []
        text, label = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    text.append(word)
                    label.append(pairs[-1])

                else:
                    if len(text) > 0:
                        texts.append(text)
                        labels.append(label)

                    text, label = [], []

        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        text_id = []
        label_id = []
        text = self.texts[item]
        label = self.labels[item]
        seq_char_list = list()

        for word in text:
            text_id.append(self.word_vocab.word_to_id(word))
        text_tensor = torch.tensor(text_id).long()
        for label_ele in label:
            label_id.append(self.label_vocab.label_to_id(label_ele))
        label_tensor = torch.tensor(label_id).long()

        for word in text:
            char_list = list(word)
            char_id = list()
            for char in char_list:
                char_id.append(self.alphabet.char_to_id(char))
            seq_char_list.append(char_id)

        return {'text': text_tensor, 'label': label_tensor, 'char': seq_char_list}
