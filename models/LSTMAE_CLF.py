import torch.nn as nn

from models.LSTMAE import LSTMAE


class LSTMAECLF(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_ratio, n_classes, seq_len, use_act=True):
        super(LSTMAECLF, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.use_act = use_act

        self.lstmae = LSTMAE(input_size=input_size, hidden_size=hidden_size, dropout_ratio=dropout_ratio, seq_len=seq_len, use_act=use_act)
        self.fc = nn.Linear(in_features=hidden_size, out_features=n_classes)

    def forward(self, x):
        rec_out, last_h = self.lstmae(x, return_last_h=True)
        out = self.fc(last_h.squeeze())
        return rec_out, out
