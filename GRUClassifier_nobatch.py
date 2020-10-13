from GRUclassifier import GRUclassifier

class GRUclassifier_nb(GRUclassifier):
    def forward(self, x):
        batch_size = 1
        h = self.init_hidden(batch_size)
        output = self.embed(x)
        output = output.view(len(x), 1, -1)

        output, hidden = self.gru(output, h)
        output = output.contiguous().view(-1, self.hidden_size * len(x)) # -1 just infers the size
        output = self.linear(output) # squish it! This is a fully connected layer!
        return self.logsoftmax(output)
