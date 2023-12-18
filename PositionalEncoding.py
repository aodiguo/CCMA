class PositionalEncoding(nn.Module):
    
    def __init__(self, num_hiddens, dropout, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        if num_hiddens%2==0:
            self.P[:, :, 1::2] = torch.cos(X)
        else:
            self.P[:, :, 1::2] = torch.cos(np.delete(X.cpu(),-1,axis=1).cuda())


    def forward(self, X):
        X = X.cuda() + self.P[:, :X.shape[1], :].cuda()
        return self.dropout(X)
