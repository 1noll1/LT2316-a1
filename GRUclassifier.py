import torch
from torch import optim
from torch import nn
criterion = nn.NLLLoss()

class GRUclassifier(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, dev):
        super().__init__()
        self.dev = dev
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size+1, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=False) #input_size is 100? 
        #self.linear = nn.Linear(100 * hidden_size, output_size) # 100 is the sequence length
        self.linear = nn.Linear(input_size * hidden_size, output_size) # 100 is the sequence length
        self.logsoftmax = nn.LogSoftmax(dim=1) # log softmax is needed for NLLL

    # def forward(self, x):
    #     output = self.embed(x)
    #     h = self.init_hidden(len(x))
    #     #h = self.init_hidden(len(x[0]))
    #     #h = self.init_hidden(len(x))
    #     output, hidden = self.gru(output, h)
    #     output = output.contiguous().view(-1, self.hidden_size * len(x[0])) # -1 just infers the size
    #     output = self.linear(output) # squish it! This is a fully connected layer!
    #     return self.logsoftmax(output)

    def forward(self, x):
        #batch_size = 1
        # h = self.init_hidden(batch_size)
        # output = self.embed(x)
        # output = output.view(len(x), 1, -1)
        # #print(output.shape)
        # output, hidden = self.gru(output, h)
        # output = output.contiguous().view(-1, self.hidden_size * len(x)) # -1 just infers the size
        # output = self.linear(output) # squish it! This is a fully connected layer!
        # return self.logsoftmax(output)

        output = self.embed(x)
        h = self.init_hidden(len(x[0]))
        #h = self.init_hidden(len(x))
        output, hidden = self.gru(output, h)
        output = output.contiguous().view(-1, self.hidden_size * len(x[0])) # -1 just infers the size
        output = self.linear(output) # squish it! This is a fully connected layer!
        return self.logsoftmax(output)

        # #h = self.init_hidden(batch_size)
        # h = self.init_hidden(len(x[0]))
        # output = self.embed(x)
        # output = output.view(len(x), 1, -1)
        # print(output.shape)
        # output, hidden = self.gru(output, h)
        # output = output.contiguous().view(-1, self.hidden_size * len(x)) # -1 just infers the size
        # output = self.linear(output) # squish it! This is a fully connected layer!
        # return self.logsoftmax(output)

    def set_dev(self, dev):
        self.dev = dev

    def init_hidden(self, x_len):
        return torch.zeros(self.num_layers, x_len, self.hidden_size).to(self.dev)
