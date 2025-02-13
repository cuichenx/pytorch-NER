# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharCNN(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout):
        super(CharCNN, self).__init__()
        print("build char sequence feature extractor: CNN ...")
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(CharCNN.random_embedding(alphabet_size, embedding_dim)))
        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)

    @staticmethod
    def random_embedding(vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        pretrain_emb[0, :] = np.zeros((1, embedding_dim))
        return pretrain_emb

    def forward(self, input):

        batch_size = input.size(0)  # (N*L, num_char)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()  #(N*L, char_emb_dim, num_char)
        char_cnn_out = self.char_cnn(char_embeds)   #(N*L, char_hid_dim, num_char)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).contiguous().view(batch_size, -1)  #(N*L, char_hid_dim)
        return char_cnn_out


class CharMLP(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, what_char):
        super().__init__()
        print("build char sequence feature extractor: MLP ...")
        # self.hidden_dim = hidden_dim
        # self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(CharCNN.random_embedding(alphabet_size, embedding_dim)))
        if what_char == 'phonemes':
            self.num_char = 3  # onset rhyme tone, for hmong
        elif what_char == 'tones':
            self.num_char = 1  # just tone
        self.mlp = nn.Linear(embedding_dim*self.num_char, hidden_dim)

    def forward(self, input):
        batch_size = input.size(0)
        char_embeds = self.char_embeddings(input)
        char_embeds = char_embeds.view(batch_size, -1)  #(N*L, char_emb_dim*num_char)
        char_cnn_out = self.mlp(char_embeds)
        return char_cnn_out