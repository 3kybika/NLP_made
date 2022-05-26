import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, heads_num, dropout, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.heads_num = heads_num

        assert hidden_dim % heads_num == 0
        self.head_dim = hidden_dim // heads_num

        self.queries_layer = nn.Linear(hidden_dim, hidden_dim)
        self.keys_layer = nn.Linear(hidden_dim, hidden_dim)
        self.values_layer = nn.Linear(hidden_dim, hidden_dim)
        self.outputs_layer = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale_factor = torch.sqrt(
            torch.FloatTensor([self.head_dim])
        ).to(device)

    def forward(self, query, key, value, mask=None):
        """
        :param query: [batch_size, query_len, hidden_dim]
        :param key:   [batch_size, key_len, hidden_dim]
        :param value: [batch_size, value len, hidden_dim]
        :output x:         [batch_size, query_len, hidden_dim]
        :output attention: [batch_size, heads_num, query_len, key_len]
        """
        batch_size = query.shape[0]

        # queries_vector = [batch_size, query_len, hidden_dim]
        # keys_vector    = [batch_size, key_len, hidden_dim]
        # values_vector  = [batch_size, value len, hidden_dim]
        queries_vector = self.queries_layer(query)
        keys_vector = self.keys_layer(key)
        values_vector = self.values_layer(value)

        # queries_vector = [batch_size, heads_num, query_len, head dim]
        # keys_vector    = [batch_size, heads_num, key_len, head dim]
        # values_vector  = [batch_size, heads_num, value len, head dim]
        queries_vector = queries_vector.view(batch_size, -1, self.heads_num, self.head_dim).permute(0, 2, 1, 3)
        keys_vector = keys_vector.view(batch_size, -1, self.heads_num, self.head_dim).permute(0, 2, 1, 3)
        values_vector = values_vector.view(batch_size, -1, self.heads_num, self.head_dim).permute(0, 2, 1, 3)

        # energy = [batch_size, heads_num, query_len, key_len]
        energy = torch.matmul(queries_vector, keys_vector.permute(0, 1, 3, 2)) / self.scale_factor

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # attention = [batch_size, heads_num, query_len, key_len]
        attention = torch.softmax(energy, dim=-1)

        # x = [batch_size, heads_num, query_len, head dim]
        x = torch.matmul(self.dropout(attention), values_vector)

        # x = [batch_size, query_len, heads_num, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch_size, query_len, hidden_dim]
        x = x.view(batch_size, -1, self.hidden_dim)

        # x = [batch_size, query_len, hidden_dim]
        x = self.outputs_layer(x)

        # x = [batch_size, query_len, hidden_dim]
        # attention = [batch_size, heads_num, query_len, key_len]
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, feed_forward_dim, dropout):
        super().__init__()

        self.feed_forward_in_layer = nn.Linear(hidden_dim, feed_forward_dim)
        self.feed_forward_out_layer = nn.Linear(feed_forward_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x:  [batch_size, seq_len, hidden_dim]
        :output x: [batch_size, seq_len, hidden_dim]
        """
        # x = [batch_size, seq_len, hidden_dim]
        x = self.dropout(torch.relu(self.feed_forward_in_layer(x)))

        # x = [batch_size, seq_len, hidden_dim]
        x = self.feed_forward_out_layer(x)

        # x = [batch_size, seq_len, hidden_dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            feed_forward_dim,
            heads_num,
            dropout,
            device
    ):
        super().__init__()

        self.attention_layer = MultiHeadAttentionLayer(hidden_dim, heads_num, dropout, device)
        self.attention_norm_layer = nn.LayerNorm(hidden_dim)

        self.feed_forward_layer = PositionwiseFeedforwardLayer(hidden_dim, feed_forward_dim, dropout)
        self.feed_forward_norm_layer = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """
        :param src:      [batch_size, src_len, hidden_dim]
        :param src_mask: [batch_size, 1, 1, src_len]
        :output src:     [batch_size, src_len, hidden_dim]
        """
        tmp_vector, _ = self.attention_layer(src, src, src, src_mask)

        # src = [batch_size, src_len, hidden_dim]
        src = self.attention_norm_layer(src + self.dropout(tmp_vector))

        # src = [batch_size, src_len, hidden_dim]
        src = self.feed_forward_norm_layer(
            src + self.dropout(self.feed_forward_layer(src))
        )

        return src


class Encoder(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            feed_forward_dim,
            layers_num,
            heads_num,
            max_length,
            dropout,
            device
    ):
        super().__init__()

        self.device = device

        self.token_embedding = nn.Embedding(input_dim, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim,
                    feed_forward_dim,
                    heads_num,
                    dropout,
                    device
                ) for _ in range(layers_num)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.scale_factor = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, src, src_mask):
        """
        :param src:      [batch_size, src_len]
        :param src_mask: [batch_size, 1, 1, src_len]
        :output src:    [batch_size, src_len, hidden_dim]
        """

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #print(pos.shape, src.shape)
        # src = [batch_size, src_len, hidden_dim]
        src = self.dropout((self.token_embedding(src) * self.scale_factor) + self.position_embedding(pos))

        for layer in self.layers:
            # src = [batch_size, src_len, hidden_dim]
            src = layer(src, src_mask)

        # src = [batch_size, src_len, hidden_dim]
        return src


class DecoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            feed_forward_dim,
            heads_num,
            dropout,
            device
    ):
        super().__init__()

        self.attention_layer = MultiHeadAttentionLayer(hidden_dim, heads_num, dropout, device)
        self.attention_norm_layer = nn.LayerNorm(hidden_dim)

        self.encoder_attention_layer = MultiHeadAttentionLayer(hidden_dim, heads_num, dropout, device)
        self.encoder_attention_norm_layer = nn.LayerNorm(hidden_dim)

        self.feed_forward_layer = PositionwiseFeedforwardLayer(hidden_dim, feed_forward_dim, dropout)
        self.feed_forward_norm_layer = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        :param trg:      = [batch_size, trg len, hidden_dim]
        :param enc_src:  = [batch_size, src_len, hidden_dim]
        :param trg_mask: = [batch_size, 1, trg len, trg len]
        :param src_mask: = [batch_size, 1, 1, src_len]
        
        :output trg:       [batch_size, trg len, hidden_dim]
        :output attention: [batch_size, heads_num, trg len, src_len]
        """

        tmp_vector, _ = self.attention_layer(trg, trg, trg, trg_mask)
        # trg = [batch_size, trg len, hidden_dim]
        trg = self.attention_norm_layer(trg + self.dropout(tmp_vector))

        # attention = [batch_size, heads_num, trg len, src_len]
        tmp_vector, attention = self.encoder_attention_layer(trg, enc_src, enc_src, src_mask)
        # trg = [batch_size, trg len, hidden_dim]
        trg = self.encoder_attention_norm_layer(trg + self.dropout(tmp_vector))

        tmp_vector = self.feed_forward_layer(trg)
        # trg = [batch_size, trg len, hidden_dim]
        trg = self.feed_forward_norm_layer(trg + self.dropout(tmp_vector))

        # trg = [batch_size, trg len, hidden_dim]
        # attention = [batch_size, heads_num, trg len, src_len]
        return trg, attention


class Decoder(nn.Module):
    def __init__(
            self,
            output_dim,
            hidden_dim,
            feed_forward_dim,
            layers_num,
            heads_num,
            max_length,
            dropout,
            device,
    ):
        super().__init__()

        self.device = device

        self.token_embedding = nn.Embedding(output_dim, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_dim,
                    feed_forward_dim,
                    heads_num,
                    dropout,
                    device
                ) for _ in range(layers_num)
            ]
        )
        self.scale_factor = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

        self.outputs_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        :param trg:      = [batch_size, trg len]
        :param enc_src:  = [batch_size, src_len, hidden_dim]
        :param trg_mask: = [batch_size, 1, trg len, trg len]
        :param src_mask: = [batch_size, 1, 1, src_len]
        
        :output output:    [batch_size, trg len, output dim]
        :output attention: [batch_size, heads_num, trg len, src_len]
        """

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # pos = [batch_size, trg len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # trg = [batch_size, trg len, hidden_dim]
        trg = self.dropout(
            (self.token_embedding(trg) * self.scale_factor) + self.position_embedding(pos)
        )

        for layer in self.layers:
            # trg = [batch_size, trg len, hidden_dim]
            # attention = [batch_size, heads_num, trg len, src_len]
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # output = [batch_size, trg len, output dim]
        output = self.outputs_layer(trg)

        # output = [batch_size, trg len, output dim]
        # attention = [batch_size, heads_num, trg len, src_len]
        return output, attention


class Seq2Seq(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            src_pad_idx,
            trg_pad_idx,
            device
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.device = device

    def get_src_vector_mask(self, src):
        """
        :param src: = [batch_size, src_len]
        """

        # src_mask = [batch_size, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def get_trg_vector_mask(self, trg):
        """
        :param trg:        [batch_size, trg len]
        :output: trg_mask: [batch_size, 1, trg len, trg len]
        :output  trg_mask: [batch_size, 1, trg len, trg len]
        """
        trg_len = trg.shape[1]

        # trg_pad_mask = [batch_size, 1, 1, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_sub_mask = [trg len, trg len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_mask = [batch_size, 1, trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch_size, 1, trg len, trg len]
        return trg_mask

    def forward(self, src, trg):
        """
        :param src: [batch_size, src_len]
        :param trg: [batch_size, trg len]
        :output output:    [batch_size, trg len, output dim]
        :output attention: [batch_size, heads_num, trg len, src_len]
        """

        # src_mask = [batch_size, 1, 1, src_len]
        src_mask = self.get_src_vector_mask(src)
        # trg_mask = [batch_size, 1, trg len, trg len]
        trg_mask = self.get_trg_vector_mask(trg)

        # enc_src = [batch_size, src_len, hidden_dim]
        enc_src = self.encoder(src, src_mask)

        # output = [batch_size, trg len, output dim]
        # attention = [batch_size, heads_num, trg len, src_len]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention
