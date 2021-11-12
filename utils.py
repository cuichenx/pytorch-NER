import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

#
# class WordVocabulary(object):
#     def __init__(self, data_path):
#         d = torch.load(data_path)
#         self.w2i = d['w2i']
#         self.i2w = d['i2w']
#
#     def unk(self):
#         return self.w2i['UNK']
#
#     def pad(self):
#         return self.w2i['PAD']
#
#     def size(self):
#         return len(self.i2w)
#
#     def word_to_id(self, word):
#         return self.w2i.get(word, self.unk())
#
#     def id_to_word(self, cur_id):
#         return self.i2w[cur_id]
#
#     def items(self):
#         return self.w2i.items()
#
#
# class LabelVocabulary(object):
#     def __init__(self, data_path):
#         d = torch.load(data_path)
#         self.l2i = d['l2i']
#         self.i2l = d['i2l']
#
#     def size(self):
#         return len(self.i2l)
#
#     def label_to_id(self, label):
#         return self.l2i[label]
#
#     def id_to_label(self, cur_id):
#         return self.i2l[cur_id]
#
#
# class Alphabet(object):
#     def __init__(self, data_path):
#         d = torch.load(data_path)
#         self.c2i = d['c2i']
#         self.i2c = d['i2c']
#         self.wi2ci = d['wi2ci']
#
#     def unk(self):
#         return self.c2i['UNK']
#
#     def pad(self):
#         return self.c2i['PAD']
#
#     def size(self):
#         return len(self.i2c)
#
#     def char_to_id(self, char):
#         return self.c2i.get(char, self.unk())
#
#     def id_to_char(self, cur_id):
#         return self.i2c[cur_id]
#
#     def items(self):
#         return self.c2i.items()


def my_collate(key, batch_tensor):
    if key == 'char':
        batch_tensor = pad_char(batch_tensor)
        return batch_tensor
    else:
        word_seq_lengths = torch.LongTensor(list(map(len, batch_tensor)))
        _, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        batch_tensor.sort(key=lambda x: len(x), reverse=True)
        tensor_length = [len(sq) for sq in batch_tensor]
        batch_tensor = pad_sequence(batch_tensor, batch_first=True, padding_value=0)
        return batch_tensor, tensor_length, word_perm_idx


def my_collate_fn(batch):
    return {key: my_collate(key, [d[key] for d in batch]) for key in batch[0]}


def pad_char(chars):
    batch_size = len(chars)
    max_seq_len = max(map(len, chars))
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len)).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    return char_seq_tensor


def load_pretrain_emb(embedding_path):
    embedd_dim = 100
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if not embedd_dim + 1 == len(tokens):
                continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


def build_pretrain_embedding(embedding_path, word_vocab, embedd_dim=100):
    embedd_dict = dict()
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    vocab_size = word_vocab.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_vocab.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_vocab.items():
        if word in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrain_emb[0, :] = np.zeros((1, embedd_dim))
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / vocab_size))
    return pretrain_emb


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_mask(batch_tensor):
    mask = batch_tensor.eq(0)
    mask = mask.eq(0)
    return mask


def write_result(s, also_print=False):
    if also_print: print(s)
    # don't need to write results, now that we're using wandb
    # with open("ALL_RESULTS.txt", 'a') as f:
    #     f.write(s)

from rpa_regex import RPA_SYLLABLE
def regex_parsable(syl):
    m = RPA_SYLLABLE.match(syl)
    if m:
        ons, rhy, ton = m.group("ons"), m.group("rhy"), m.group("ton")
        return ons + rhy + ton == syl
    return False

def hmong_syllable_component(syl, component=None):
    m = RPA_SYLLABLE.match(syl)
    if m:
        ons, rhy, ton = m.group("ons"), m.group("rhy"), m.group("ton")
        ret = {'ons': ons, 'rhy': rhy, 'ton': ton}
        if component is None:
            return ret
        else:
            return ret.get(component, None)

