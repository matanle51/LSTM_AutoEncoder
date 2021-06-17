# LSTM Auto-Encoder (LSTM-AE) implementation in Pytorch
The code implements three variants of LSTM-AE:
1. Regular LSTM-AE for reconstruction tasks (LSTMAE.py)
2. LSTM-AE + Classification layer after the decoder (LSTMAE_CLF.py)
3. LSTM-AE + prediction layer on top of the encoder (LSTMAE_PRED.py)

To test the implementation, we defined three different tasks:

Toy example (on random uniform data) for sequence reconstruction:
```
python lstm_ae_toy.py
```

MNIST reconstruction + classification:
```
python lstm_ae_mnist.py
```

SnP stock daily graph reconstruction + price prediction:
```
python lstm_ae_snp500.py
```