import torch.nn as nn

from models.LSTMAE import LSTMAE


class LSTMAEPRED(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_ratio, seq_len, use_act=True):
        super(LSTMAEPRED, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.seq_len = seq_len
        self.use_act = use_act  # Parameter to control the last sigmoid activation - depends on the normalization used.

        # AE
        self.lstmae = LSTMAE(input_size=input_size, hidden_size=hidden_size, dropout_ratio=dropout_ratio, seq_len=seq_len, use_act=use_act)

        # Prediction layers
        self.pred_fc = nn.Linear(hidden_size, input_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Calculate AE reconstruction
        rec_out, enc_out = self.lstmae(x, return_enc_out=True)

        # Calculate Prediction sequence
        if self.use_act:
            pred_out = self.sig(self.pred_fc(enc_out)).squeeze()
        else:
            pred_out = self.pred_fc(enc_out).squeeze()

        return rec_out, pred_out
