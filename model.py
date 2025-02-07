import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from crf import CRF
from charcnn import CharCNN, CharMLP


class NamedEntityRecog(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, word_hidden_dim, alphabet_size, char_embedding_dim, char_hidden_dim,
                 feature_extractor, tag_num, dropout, pretrain_embed=None, use_char=False, use_crf=False, use_gpu=False,
                 char_feature_extractor=None, what_char='phonemes', char_after_cnn=False):
        super(NamedEntityRecog, self).__init__()
        self.use_crf = use_crf
        self.use_char = use_char
        self.drop = nn.Dropout(dropout)
        self.input_dim = word_embed_dim
        self.feature_extractor = feature_extractor
        self.char_after_cnn = char_after_cnn
        if word_embed_dim > 0:
            self.embeds = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
            if pretrain_embed is not None:
                self.embeds.weight.data.copy_(torch.from_numpy(pretrain_embed))
            else:
                self.embeds.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, word_embed_dim)))
        else:
            self.embeds = None

        if self.use_char:
            if not char_after_cnn:
                self.input_dim += char_hidden_dim
            if char_feature_extractor == 'cnn':
                self.char_feature = CharCNN(alphabet_size, char_embedding_dim, char_hidden_dim, dropout)
            else:
                self.char_feature = CharMLP(alphabet_size, char_embedding_dim, char_hidden_dim, what_char)

        if feature_extractor == 'lstm':
            self.lstm = nn.LSTM(self.input_dim, word_hidden_dim, batch_first=True, bidirectional=True)
        else:
            self.word2cnn = nn.Linear(self.input_dim, word_hidden_dim*2)
            self.cnn_list = list()
            for _ in range(4):
                self.cnn_list.append(nn.Conv1d(word_hidden_dim*2, word_hidden_dim*2, kernel_size=3, padding=1))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(dropout))
                self.cnn_list.append(nn.BatchNorm1d(word_hidden_dim*2))
            self.cnn = nn.Sequential(*self.cnn_list)

        if self.use_crf:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num + 2)
            self.crf = CRF(tag_num, use_gpu)
        elif char_after_cnn:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2 + char_hidden_dim, tag_num)
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward_common(self, word_inputs, word_seq_lengths, char_inputs):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        word_list = [self.embeds(word_inputs)] if self.embeds is not None else []

        if self.use_char:
            char_features = self.char_feature(char_inputs).contiguous().view(batch_size, seq_len, -1)
            if not self.char_after_cnn:
                word_list.append(char_features)
        word_embedding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embedding) if self.embeds is not None else word_embedding
        if self.feature_extractor == 'lstm':
            packed_words = pack_padded_sequence(word_represents, word_seq_lengths, True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            lstm_out = lstm_out.transpose(0, 1)
            feature_out = self.drop(lstm_out)
        else:
            word_in = torch.tanh(self.word2cnn(word_represents)).transpose(2, 1).contiguous()
            feature_out = self.cnn(word_in).transpose(1, 2).contiguous()

        if self.use_char and self.char_after_cnn:
            feature_out = torch.cat((feature_out, char_features), dim=2)

        logits = self.hidden2tag(feature_out)
        return logits

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        feature_out = self.forward_common(word_inputs, word_seq_lengths, char_inputs)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label)
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            total_loss = loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len))
        return total_loss

    def forward(self, word_inputs, word_seq_lengths, char_inputs, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        logits = self.forward_common(word_inputs, word_seq_lengths, char_inputs)

        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(logits, mask)
        else:
            logits = logits.contiguous().view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(logits, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
        return tag_seq
