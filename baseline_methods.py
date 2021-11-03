import os, codecs

import gensim
import torch
from torch.utils.data import DataLoader

from dataset import SCH_ElaborateExpressions
from rpa_regex import RPA_SYLLABLE

# two baseline methods
# 1. every ABAC is an EE (should have 100% recall, low precision)
# 2. every ABAC with w2v_sim(B, C) > thresh is an EE (should perform better)
from utils import my_collate_fn


class BaselineModel:
    def __init__(self, i2w, l2i, w2v_model_path):
        self.i2w = i2w
        self.l2i = l2i
        self.syl_regex = RPA_SYLLABLE
        self.w2v_model = gensim.models.Word2Vec.load(w2v_model_path)

    def regex_parsable(self, syl):
        if type(syl) == int:
            syl = self.i2w[syl]
        m = self.syl_regex.match(syl)
        if m:
            ons, rhy, ton = m.group("ons"), m.group("rhy"), m.group("ton")
            return ons + rhy + ton == syl
        return False

    def w2v_sim(self, w1, w2):
        if type(w1) == int:
            w1 = self.i2w[w1]
        if type(w2) == int:
            w2 = self.i2w[w2]
        try:
            return self.w2v_model.wv.similarity(w1, w2)
        except KeyError:
            return 0


    def __call__(self, batch_text, sim_thresh=-1, ensure_A_parsable=False):
        # text of shape (N, L)
        # we're going to do an offset difference of text with itself
        # if there is a zero in the offset difference, then we've found the 'A' word
        # the words after it are the B and C words
        # Note that we need to change the padding of one of them (to -1) to make sure we don't get any matches on the
        # padding characters

        # text1 = text[:, 2:]
        # text2 = torch.where(text!=0, text, -torch.ones_like(text))[:, :-2]
        # offset_diff = text1 - text2
        # A_positions = torch.where(offset_diff==0)
        word_PAD = 0 # self.word_vocab.word_to_id('PAD')
        tag_PAD = self.l2i['PAD']
        tag_B = self.l2i['B']
        tag_I = self.l2i['I']
        tag_O = self.l2i['O']

        tag_seq = []
        for text in batch_text:
            # if text is PAD, then tag is PAD, else tag is O
            text = text.tolist()
            tag_seq.append([tag_PAD if t == word_PAD else tag_O for t in text])
            i = 0
            while i < len(text)-3 and text[i] != word_PAD:
                if text[i] == text[i+2]:
                    A = text[i]
                    if ensure_A_parsable and not self.regex_parsable(A):
                        i += 1
                        continue
                    B = text[i+1]
                    C = text[i+3]
                    if sim_thresh == -1 or self.w2v_sim(B, C) > sim_thresh:
                        tag_seq[-1][i] = tag_B
                        tag_seq[-1][i+1] = tag_I
                        tag_seq[-1][i+2] = tag_I
                        tag_seq[-1][i+3] = tag_I
                        i += 4
                        continue
                i += 1
        return tag_seq


def evaluate_baseline(dataloader, model, i2w, i2l, pred_file, score_file, eval_script,
                      sim_thresh=-1, ensure_A_parsable=False):
    prediction = []
    for batch in dataloader:
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']

        tag_seq = model(batch_text, sim_thresh=sim_thresh, ensure_A_parsable=ensure_A_parsable)

        for line_tesor, labels_tensor, predicts_list in zip(batch_text, batch_label, tag_seq):
            for word_tensor, label_tensor, predict in zip(line_tesor, labels_tensor, predicts_list):
                if word_tensor.item() == 0:
                    break
                line = ' '.join(
                    [i2w[word_tensor.item()], i2l[label_tensor.item()], i2l[predict]])
                prediction.append(line)
            prediction.append('')

    with open(pred_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, pred_file, score_file))

    eval_lines = [l.rstrip() for l in codecs.open(score_file, 'r', 'utf8')]
    new_f1 = -1

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_f1 = float(line.strip().split()[-1])
            break

    return new_f1


if __name__ == '__main__':
    split = torch.load("data/split_grouped_3.pth")
    test_dataset = SCH_ElaborateExpressions("data/data.pth", split['test'], positive_ratio=-1, switch_prob=0, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)

    w2v_model_path = "/home/cuichenx/Research/elab-order/data/hmong/sch.sg.w2v"
    model = BaselineModel(test_dataset.i2w, test_dataset.l2i, w2v_model_path)

    eval_path = "evaluation"
    eval_temp = os.path.join(eval_path, "temp")
    eval_script = os.path.join(eval_path, "conlleval")
    pred_file = eval_temp + '/pred.txt'
    score_file = eval_temp + '/score.txt'

    for sim_thresh in (-1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,):
        for ensure_A_parsable in (False, True):
            print("-"*50)
            print("sim_thresh:", sim_thresh, "\tensure_A_parsable:", ensure_A_parsable)
            evaluate_baseline(test_dataloader, model, test_dataset.i2w, test_dataset.i2l, pred_file, score_file, eval_script,
                              sim_thresh=sim_thresh, ensure_A_parsable=ensure_A_parsable)
            break
        break
