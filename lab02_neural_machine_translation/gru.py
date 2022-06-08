class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, layers_num, max_length, dropout_p):
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = GeneralAttn(hidden_dim)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, layers_num, dropout=dropout_p)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        
        embed_vector = self.dropout(self.embedding(word_input).view(1, 1, -1))
        
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        
        output, hidden = self.gru(
            torch.cat((embed_vector, context), 2), 
            last_hidden
        )
        
        output = output.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        
        return output, hidden, attn_weights

