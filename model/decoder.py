from torch import nn 

class Map2Seq(nn.Module):
    def __init__(self):
        super(Map2Seq, self).__init__()

    def forward(self, x):
        '''
            x: (bs, c, h, w)
        '''
        bs, c, h, w = x.shape
        x = x.contiguous().view(bs, c*h, w)
        x = x.permute(2, 0, 1) # (w, bs, dim)

        return x

class RecurrentBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RecurrentBlock, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)

        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(Decoder, self).__init__()

        self.recurrent_blocks = nn.Sequential(
            RecurrentBlock(input_dim, hidden_dim),
            RecurrentBlock(hidden_dim*2, hidden_dim),
            RecurrentBlock(hidden_dim*2, hidden_dim),
            RecurrentBlock(hidden_dim*2, hidden_dim),
            RecurrentBlock(hidden_dim*2, hidden_dim)
        )

        self.generator = nn.Sequential(
            nn.Linear(2*hidden_dim, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        '''
            x: (w, bs, dim)
        ''' 
        x = self.recurrent_blocks(x) # (w, bs, dim)

        return self.generator(x) # (w, bs, vocab_size)