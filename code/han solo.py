import torch
import torch.nn as nn

class HAN_SOLO(nn.Module):
    def __init__(self, embed_size, gru_hidden_size, device='cpu'):
        super(HAN_SOLO, self).__init__()
        self.embed_size = embed_size
        self.gru_hidden_size = gru_hidden_size

        # word gru
        self.biGRU_word = nn.GRU(embed_size, gru_hidden_size, bidirectional=True)
        # word attention
        self.linear_word = nn.Linear(gru_hidden_size * 2, gru_hidden_size * 2)
        self.tanh_word = nn.Tanh()
        self.context_word = nn.Parameter(gru_hidden_size * 2, 1)
        self.softmax_word = nn.Softmax()
        # sentence gru
        self.biGRU_sent = nn.GRU(embed_size, gru_hidden_size, bidirectional=True)
        # sentence attention
        self.linear_sentence = nn.Linear(gru_hidden_size * 2, gru_hidden_size * 2)
        self.tanh_sentence = nn.Tanh()
        self.context_sentence = nn.Parameter(gru_hidden_size * 2, 1)
        self.softmax_sentence = nn.Softmax()

    def forward(self, x):

        # word gru
        f_output_word, h_output_word = self.biGRU_word(x)
        # word attention
        word_lin_out = self.linear_word(f_output_word)
        word_tanh_out = self.tanh_word(word_lin_out)
        softmax_input_word = torch.mm(word_tanh_out, self.context_word)
        alpha_word = self.softmax_word(softmax_input_word)
        S = torch.dot(alpha_word, f_output_word) # it might be a *
        # sentence gru
        f_output_sentence, h_output_sentence = self.biGRU_sent(S)
        # sentence attention
        sentence_lin_out = self.linear_sentence(f_output_sentence)
        sentence_tanh_out = self.tanh_sentence(sentence_lin_out)
        softmax_input_sentence = torch.mm(sentence_tanh_out, self.context_sentence)
        alpha_sentence = self.softmax_sentence(softmax_input_sentence)
        V = torch.dot(alpha_sentence, f_output_sentence)  # it might be a *

        return V

# main for testing the class
from train import *
embed_size = 300
gru_hidden_size = 50

