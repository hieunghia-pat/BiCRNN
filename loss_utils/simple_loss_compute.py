class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, src_len, tgt_len):
        '''
            x: (w, bs, vocab_size)
            y: (bs, max_len)
            src_len: (bs, )
            tgt_len: (bs, )
        '''
        loss = self.criterion(x, y, src_len, tgt_len)
        if self.opt is not None:
            # self.opt.optimizer.zero_grad()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss