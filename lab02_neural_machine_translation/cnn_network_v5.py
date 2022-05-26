import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
            self,
            input_dim,
            embedding_dim,
            hidden_dim,
            layers_num,
            kernel_size,
            max_length,
            dropout,
            device
    ):
        super().__init__()

        self.device = device
        self.scale_factor = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(input_dim, embedding_dim)
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)

        self.embedding_to_hidden_layer = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_to_embedding_layer = nn.Linear(hidden_dim, embedding_dim)

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=2 * hidden_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                )
                for _ in range(layers_num)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        :param src:              [batch_size, src_len]
        :return: cur_conv_vector [batch_size, src_len, embbeding_dim]
        :return: result_vector   [batch_size, src_len, embbeding_dim]
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = [batch_size, src_len]
        pos = (
            torch.arange(0, src_len)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .to(self.device)
        )

        # embed_vector = [batch_size, src_len, embbeding_dim]
        embed_vector = self.dropout(
            self.tok_embedding(src) + self.pos_embedding(pos)
        )

        # conv_input = [batch_size, hidden_dim, src_len]
        conv_prev_vector = self.embedding_to_hidden_layer(embed_vector).permute(0, 2, 1)

        for i, conv_layer in enumerate(self.conv_layers):
            # cur_conv_vector = [batch_size, 2 * hidden_dim, src_len]
            cur_conv_vector = conv_layer(self.dropout(conv_prev_vector))

            # cur_conv_vector = [batch_size, hidden_dim, src_len]
            cur_conv_vector = F.glu(cur_conv_vector, dim=1)

            # cur_conv_vector = [batch_size, hidden_dim, src_len]
            cur_conv_vector = (cur_conv_vector + conv_prev_vector) * self.scale_factor

            conv_prev_vector = cur_conv_vector

        # conved = [batch_size, src_len, embbeding_dim]
        cur_conv_vector = self.hidden_to_embedding_layer(cur_conv_vector.permute(0, 2, 1))

        # result_vector = [batch_size, src_len, embbeding_dim]
        result_vector = (cur_conv_vector + embed_vector) * self.scale_factor

        # cur_conv_vector = [batch_size, src_len, embbeding_dim]
        # result_vector = [batch_size, src_len, embbeding_dim]
        return cur_conv_vector, result_vector


class Decoder(nn.Module):
    def __init__(
            self,
            output_dim,
            embedding_dim,
            hidden_dim,
            layers_num,
            kernel_size,
            trg_pad_idx,
            max_length,
            dropout,
            device,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.scale_factor = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, embedding_dim)
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)

        self.embedding_to_hidden_layer = nn.Linear(embedding_dim, hidden_dim)
        self.hidden_to_embedding_layer = nn.Linear(hidden_dim, embedding_dim)

        self.attn_hidden_to_embedding_layer = nn.Linear(hidden_dim, embedding_dim)
        self.attn_embedding_to_hidden_layer = nn.Linear(embedding_dim, hidden_dim)

        self.output_layer = nn.Linear(embedding_dim, output_dim)

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=2 * hidden_dim,
                    kernel_size=kernel_size
                ) for _ in range(layers_num)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def get_attention(self, embedded_vector, conved_vector, encoder_conved, encoder_combined):
        """
        :param embedded_vector:  [batch_size, trg_len, embbeding_dim]
        :param conved_vector:    [batch_size, hidden_dim, trg_len]
        :param encoder_conved:   [batch_size, src_len, embbeding_dim]
        :param encoder_combined: [batch_size, src_len, embbeding_dim]
        :return: attention_vector [batch_size, trg_len, src_len]
        :return: attended_result_vector = [batch_size, hidden_dim, trg_len]
        """

        # combined_vector = [batch_size, trg_len, embbeding_dim]
        combined_vector = (
                                  self.attn_hidden_to_embedding_layer(conved_vector.permute(0, 2, 1)) +
                                  embedded_vector
                          ) * self.scale_factor

        # energy = [batch_size, trg_len, src_len]
        energy = torch.matmul(combined_vector, encoder_conved.permute(0, 2, 1))

        # attention_vector = [batch_size, trg_len, src_len]
        attention_vector = F.softmax(energy, dim=2)

        # attended_encoding_vector = [batch_size, trg_len, emd dim]
        attended_encoding_vector = torch.matmul(attention_vector, encoder_combined)

        # attended_encoding_vector = [batch_size, trg_len, hidden_dim]
        attended_encoding_vector = self.attn_embedding_to_hidden_layer(attended_encoding_vector)

        # attended_result_vector = [batch_size, hidden_dim, trg_len]
        attended_result_vector = (
                                         conved_vector +
                                         attended_encoding_vector.permute(0, 2, 1)
                                 ) * self.scale_factor

        # attention_vector = [batch_size, trg_len, src_len]
        # attended_result_vector = [batch_size, hidden_dim, trg_len]
        return attention_vector, attended_result_vector

    def forward(self, trg, encoder_conved, encoder_combined):
        """
        :param: trg              [batch_size, trg_len]
        :param: encoder_conved   [batch_size, src_len, embbeding_dim]
        :param: encoder_combined [batch_size, src_len, embbeding_dim]
        :return: attention [batch_size, trg_len, src_len]
        :return: output    [batch_size, trg_len, output_dim]
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # pos = [batch_size, trg_len]
        pos = (
            torch
                .arange(0, trg_len)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .to(self.device)
        )

        # embedded = [batch_size, trg_len, embbeding_dim]
        embedded_vector = self.dropout(self.tok_embedding(trg) + self.pos_embedding(pos))

        # hidden_vector = [batch_size, trg_len, hidden_dim]
        hidden_vector = self.embedding_to_hidden_layer(embedded_vector)

        # conv_prev_vector = [batch_size, hidden_dim, trg_len]
        conv_prev_vector = hidden_vector.permute(0, 2, 1)

        batch_size = conv_prev_vector.shape[0]
        hidden_dim = conv_prev_vector.shape[1]

        for i, conv in enumerate(self.conv_layers):
            conv_prev_vector = self.dropout(conv_prev_vector)

            padding = torch.zeros(
                batch_size,
                hidden_dim,
                self.kernel_size - 1
            ).fill_(self.trg_pad_idx).to(self.device)

            # padded_vector = [batch_size, hidden_dim, trg_len + kernel_size - 1]
            padded_vector = torch.cat((padding, conv_prev_vector), dim=2)

            # attention = [batch_size, trg_len, src_len]
            # conv_res_vector = [batch_size, hidden_dim, trg_len]
            attention, conv_res_vector = self.get_attention(
                embedded_vector,
                F.glu(conv(padded_vector), dim=1),
                encoder_conved,
                encoder_combined
            )

            # conv_res_vector = [batch_size, hidden_dim, trg_len]
            conv_res_vector = (conv_res_vector + conv_prev_vector) * self.scale_factor

            conv_prev_vector = conv_res_vector

        # conved = [batch_size, trg_len, embbeding_dim]
        conv_res_vector = self.hidden_to_embedding_layer(conv_res_vector.permute(0, 2, 1))

        # output = [batch_size, trg_len, output_dim]
        output_vector = self.output_layer(self.dropout(conv_res_vector))
        
        # output = [batch_size, trg_len, output_dim]
        # attention = [batch_size, trg_len, src_len]
        return output_vector, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        """
        :param: src [batch_size, src_len]
        :param: trg [batch_size, trg_len - 1] 
        :return: output    [batch_size, trg_len - 1, output_dim]
        :return: attention [batch_size, trg_len - 1, src_len]
        """
        # encoder_conved = [batch_size, src_len, embbeding_dim]
        # encoder_combined = [batch_size, src_len, embbeding_dim]
        encoder_conved, encoder_combined = self.encoder(src)

        # output = [batch_size, trg_len - 1, output_dim]
        # attention = [batch_size, trg_len - 1, src_len]
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        return output, attention
