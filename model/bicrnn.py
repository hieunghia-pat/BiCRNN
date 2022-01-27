import torch
from torch import nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.decoder import Map2Seq
import config

class BiCRNN(nn.Module):
    def __init__(self, imgChannels, hidden_dim, vocab_size):
        super(BiCRNN, self).__init__()

        self.encoder = Encoder(imgChannels)
        self.map2seq = Map2Seq()
        self.decoder = Decoder(160, hidden_dim, vocab_size)

    def get_predictions(self, images):
        encoded_features = self.map2seq(self.encoder(images))
        ys = self.decoder(encoded_features).argmax(dim=-1)

        return ys

    def forward(self, x):
        '''
            x: (bs, c, image_h, image_w)
        '''
        x = self.encoder(x) # (bs, channels, h, w)
        x = self.map2seq(x) # (w, bs, dim)
        x = self.decoder(x) # (w, bs, vocab_size)
        src_len, batch_size = x.shape[:2]

        return x, torch.tensor([src_len]*batch_size, device=x.device)