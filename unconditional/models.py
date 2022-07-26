# -*- coding: utf-8 -*-
"""Model definitions of Onoma-to-Wave model (NOT conditioned on sound events).

Copyright (C) 2022 by Akira TAMAMORI
Copyright (C) 2021 Ryuichi YAMAMOTO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import random

import torch
from torch import nn


# borrowed from https://github.com/r9y9/ttslearn/blob/master/ttslearn/util.py
def _make_pad_mask(lengths, maxlen=None):
    """Make mask for padding frames.

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length. If None, use max value of lengths.
    Returns:
        torch.ByteTensor: mask
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    if maxlen is None:
        maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(int(len(lengths)), maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


class Encoder(nn.Module):
    """Encoder Class."""

    def __init__(self, dimensions, char2id, n_layers=1):
        """Initialize class.

        input_dim: number of phonemes
        emb_dim: Embedded dimension = number of dimensions of LSTM input vectors
        hidden_dim: number of dimensions of LSTM output vectors
        n_layers: number of hidden layers
        """
        super().__init__()
        input_dim = dimensions[0]
        emb_dim = dimensions[1]
        hidden_dim = dimensions[2]

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=char2id[" "])

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, source):
        """Forward propagation.

        source = [batch_size, source_len]
        """
        # embedded = [batch_size, source_len, emb_dim]
        embedded = self.embedding(source)

        # outputs = [batch_size, source_len, hidden_dim * 2]
        # hidden = [n_layers * 2, batch size, hidden_dim]
        # cell = [n_layers * 2, batch size, hidden_dim]
        _, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


class Decoder(nn.Module):
    """Decoder Class."""

    def __init__(self, output_dim, hidden_dim, n_layers=2):
        """Initialize class.

        input_dim: number of spectrogram dims. at current frame
        output_dim: number of spectrogram dims. at next frame
        hidden_dim: number of dimensions of LSTM output vectors
        n_layers: number of hidden layers
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        input_dim = output_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, encoder_state):
        """Forward propagation.

        input: spectrogram -> [batch_size, input_dim]
        encoder_state: encoder hidden state, encoder cell (, event_vector)
        """
        # input = [batch_size, 1, input_dim]
        # output = [batch_size, 1, hidden_dim]
        # hidden = [n_layers * 2, batch_size, hidden_dim]
        # cell = [n_layers * 2, batch_size, hidden_dim]
        output, (hidden, cell) = self.rnn(inputs, encoder_state)

        # prediction = [batch_size, output_dim]
        prediction = self.fc_out(output.squeeze(1))

        return prediction.unsqueeze(1), (hidden, cell)


class Seq2Seq(nn.Module):
    """Seq2Seq Class Module."""

    def __init__(self, encoder, decoder, device):
        """Initialize class."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.criterion = nn.L1Loss()

    def forward(self, onomatopes, bos_embeddings, n_frame):
        """Forward propagation."""
        encoder_state = self.encoder(onomatopes)
        inputs = bos_embeddings  # <BOS> = [batch_size, 1, hiddendim]
        n_batch = bos_embeddings.shape[0]
        n_fbin = self.decoder.output_dim
        outputs = torch.zeros(n_batch, n_frame, n_fbin, device=self.device)

        for frame in range(n_frame):
            output, encoder_state = self.decoder(inputs, encoder_state)
            outputs[:, frame : frame + 1, :] = output
            inputs = output

        return outputs

    def get_loss(self, source, target, frame_lengths, teacher_forcing_ratio=0.6):
        """Compute loss in training."""
        target_len = target.shape[1]
        target_dim = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(
            target.shape[0], target_len, target_dim, device=self.device
        )

        # last hidden state of the encoder is used
        # as the initial hidden state of the decoder
        encoder_state = self.encoder(source)

        # first input to the decoder is the <BOS> tokens
        inputs = target[:, 0:1, :]  # [batch_size, 1, output_dim]

        for frame in range(1, target_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # Compute decoder output for one frame
            if frame == 1:
                output, hidden_state = self.decoder(inputs, encoder_state)
            else:
                output, hidden_state = self.decoder(inputs, hidden_state)

            outputs[:, frame : frame + 1, :] = output  # [batch_size, 1, output_dim]

            # teacher forcing (scheduled sampling)
            teacher_force = random.random() < teacher_forcing_ratio

            # Decides whether to use the spectrogram generated by the decoder
            # as input at the next frame or the spectrogram of the target,
            # according to the teacher forcing rate
            inputs = (
                target[:, frame : frame + 1, :] if teacher_force else output.detach()
            )

        # mask = [batch_size, target_len, 1]
        mask = _make_pad_mask(frame_lengths).unsqueeze(-1).to(self.device)
        mask = ~mask  # True <-> False
        outputs = outputs[:, 1:, :].masked_select(mask)
        target = target[:, 1:, :].masked_select(mask)

        return self.criterion(outputs, target)


class BosEmbedding(nn.Module):
    """Compute embeddings of a special start charachter."""

    def __init__(self, vocab_size, embedding_dim, char2id, device):
        """Initialize class."""
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=char2id["<BOS>"]
        )
        self.char2id = char2id
        self.device = device

    def forward(self, bos_tensor):
        """Embed start character."""
        output = self.word_embeddings(bos_tensor)
        return output

    def bos_embedding(self, n_batch=1):
        """Return embeddings of <BOS>."""
        bos = torch.tensor(
            [[self.char2id["<BOS>"]] for _ in range(n_batch)], device=self.device
        )
        embedding = self.forward(bos)

        return embedding


def get_model(mapping_dict, cfg):
    """Instantiate models."""
    char2id = mapping_dict.char2id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = {"seq2seq": None, "bos": None}

    output_dim = cfg.feature.n_fft // 2 + 1
    vocab_size = len(char2id)

    encoder = Encoder(
        dimensions=(vocab_size, cfg.model.hidden_dim, cfg.model.hidden_dim),
        char2id=char2id,
        n_layers=cfg.model.n_layers_enc,
    )

    decoder = Decoder(output_dim, cfg.model.hidden_dim, n_layers=cfg.model.n_layers_dec)

    model["seq2seq"] = Seq2Seq(encoder, decoder, device).to(device)
    model["bos"] = BosEmbedding(vocab_size, output_dim, char2id, device).to(device)

    return model
