from torch.utils.data import Dataset
from utils import normalize_word
import torch
import random

class SCH_ElaborateExpressions(Dataset):
    def __init__(self, data_path, sent_ids, positive_ratio=0.5):
        self.sentences, self.tags = {}, {}
        self.l2i, self.i2l = {}, []
        self.w2i, self.i2w = {}, []
        # d has keys:
        # sentences: dict with 776615 keys (all sentences)
        # tags: dict with 23234 keys (only sentences with tagged EEs)
        # l2i: dict, i2l: list, length 3 (OIB)
        # w2i: dict, i2w: list, length 12939 (vocab size)
        self.__dict__.update(torch.load(data_path))
        self.sentences = {idx: self.sentences[idx] for idx in sent_ids}  # filer only selected data
        self.tags = {idx: self.tags[idx] for idx in sent_ids if idx in self.tags}  # filer only selected data

        # positive ratio: how many positive sentences to include in the whole usable dataset
        assert 0 < positive_ratio <= 1.0
        self.positive_length = len(self.tags)
        self.tot_length = int(self.positive_length / positive_ratio)
        self.positive_keys = list(self.tags.keys())
        self.negative_keys = list(self.sentences.keys() - self.tags.keys())


    def __len__(self):
        return self.tot_length


    def __getitem__(self, idx):
        if idx >= self.positive_length:
            sent_id = random.choice(self.negative_keys)
        else:
            sent_id = self.positive_keys[idx]
        sentence = self.sentences[sent_id]
        tag = self.tags.get(sent_id, [self.l2i['O']] * len(sentence))
        seq_char_list = [[0]] * len(sentence)  ## TODO: char level model for Hmong
        if len(tag) == 0:
            print(sent_id, 'is 0!')
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
