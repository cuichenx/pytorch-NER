import utils
from utils import get_mask
import os
import codecs
import torch
from torch.nn.utils import clip_grad_norm_
import wandb

def train_model(dataloader, model, optimizer, batch_num, writer, use_gpu=False):
    model.train()
    for batch in dataloader:
        batch_num += 1
        model.zero_grad()
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        char_inputs = batch['char']
        char_inputs = char_inputs[word_perm_idx]
        char_dim = char_inputs.size(-1)
        char_inputs = char_inputs.contiguous().view(-1, char_dim)
        if use_gpu:
            batch_text = batch_text.cuda()
            batch_label = batch_label.cuda()
            char_inputs = char_inputs.cuda()
        mask = get_mask(batch_text)
        loss = model.neg_log_likelihood_loss(batch_text, seq_length, char_inputs, batch_label, mask)
        # writer.add_scalar('loss', loss, batch_num)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    return batch_num


def evaluate(dataloader, model, word_vocab, label_vocab, pred_file, score_file, eval_script, use_gpu=False, prefix=''):
    model.eval()
    prediction = []
    for batch in dataloader:
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        char_inputs = batch['char']
        char_inputs = char_inputs[word_perm_idx]
        char_dim = char_inputs.size(-1)
        char_inputs = char_inputs.contiguous().view(-1, char_dim)
        if use_gpu:
            batch_text = batch_text.cuda()
            batch_label = batch_label.cuda()
            char_inputs = char_inputs.cuda()
        mask = get_mask(batch_text)
        with torch.no_grad():
            tag_seq = model(batch_text, seq_length, char_inputs, batch_label, mask)

        for line_tesor, labels_tensor, predicts_tensor in zip(batch_text, batch_label, tag_seq):
            for word_tensor, label_tensor, predict_tensor in zip(line_tesor, labels_tensor, predicts_tensor):
                if word_tensor.item() == 0:
                    break
                line = ' '.join(
                    [word_vocab.id_to_word(word_tensor.item()), label_vocab.id_to_label(label_tensor.item()),
                     label_vocab.id_to_label(predict_tensor.item())])
                prediction.append(line)
            prediction.append('')

    with open(pred_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, pred_file, score_file))

    eval_lines = [l.rstrip() for l in codecs.open(score_file, 'r', 'utf8')]
    new_f1 = -1
    res_lines = []
    to_log = {}

    for i, line in enumerate(eval_lines):
        print(line)
        res_lines.append(line)
        if i == 1:
            new_f1 = float(line.strip().split()[-1])
            metrics = line.split(';')
            for m in metrics:
                metric_name, num = m.split(":")
                metric_name = metric_name.strip()
                num = float(num.strip().strip("%"))
                to_log[prefix+metric_name] = num
            break

    if prefix:
        wandb.log(to_log)
    utils.write_result('\n'.join(res_lines))

#  accuracy:  99.80%; precision:  82.09%; recall:  99.01%; FB1:  89.76
    return new_f1
