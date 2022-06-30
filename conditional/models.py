# -*- coding: utf-8 -*-
"""Model class definitions of Onoma-to-Wave model (conditioned on sound events).

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
    bs = int(len(lengths))
    if maxlen is None:
        maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


def _get_lossfn_by_name(name: str):
    """Get loss function by name (string)."""
    criterion = {
        "L1": nn.L1Loss(),
        "L2": nn.MSELoss(),
        "smooth_L1": nn.SmoothL1Loss(),
    }

    if name not in criterion:
        raise ValueError(name, "is not a valid loss function.")

    return criterion[name]


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
        # hidden = [batch size, n_layers * 2, hid dim]
        # cell = [batch size, n_layers * 2, hid dim]
        _, (hidden, cell) = self.rnn(embedded)

        return hidden, cell


class Decoder(nn.Module):
    """Decoder Class."""

    def __init__(self, input_dim, output_dim, hidden_dim, n_layers=2):
        """Initialize class.

        input_dim: number of phonemes (dim. of onehot vector)
        output_dim: number of spectrogram dimensions (= number of freq. bins)
        hidden_dim: number of dimensions of LSTM output vectors
        n_layers: number of hidden layers
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, encoder_state):
        """Forward propagation.

        input: spectrogram -> [batch_size, input_dim]
        encoder_state: encoder hidden state, encoder cell (, event_vector)
        """
        # input = [batch_size, 1, input_dim]
        # output = [batch_size, 1, hidden_dim]
        # hidden = [batch_size, n_layers * 2, hidden_dim]
        # cell = [batch_size, n_layers * 2, hidden_dim]
        output, (hidden, cell) = self.rnn(inputs, encoder_state)

        # prediction = [batch_size, output_dim]
        prediction = self.fc_out(output.squeeze(1))

        return prediction.unsqueeze(1), (hidden, cell)


class Seq2Seq(nn.Module):
    """Seq2Seq Class Module."""

    def __init__(self, model, device, criterion="L1", teacher_forcing_ratio=0.6):
        """Initialize class."""
        super().__init__()

        self.encoder = model["encoder"]
        self.decoder = model["decoder"]
        self.event = model["event"]
        self.device = device
        self.criterion = _get_lossfn_by_name(criterion)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, onomatopes, bos_embeddings, event_label, n_frame):
        """Forward propagation."""
        encoder_state = self.encoder(onomatopes)
        inputs = bos_embeddings  # <BOS> = [batch_size, 1, hiddendim]
        n_batch = bos_embeddings.shape[0]
        n_fbin = self.decoder.output_dim
        outputs = torch.zeros(n_batch, n_frame, n_fbin, device=self.device)
        decoder_hidden = self.event(event_label, encoder_state)

        for frame in range(n_frame):
            if frame == 0:
                output, encoder_state = self.decoder(inputs, decoder_hidden)
            else:
                output, encoder_state = self.decoder(inputs, encoder_state)

            outputs[:, frame : frame + 1, :] = output
            inputs = output

        return outputs

    def get_loss(self, source, target, frame_lengths, event_label):
        """Compute loss in training."""
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_dim = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_dim, device=self.device)

        # last hidden state of the encoder is used
        # as the initial hidden state of the decoder
        encoder_state = self.encoder(source)

        # first input to the decoder is the <BOS> tokens
        inputs = target[:, 0:1, :]  # [batch_size, 1, output_dim]

        # Embedded event label
        decoder_hidden = self.event(event_label, encoder_state)

        for frame in range(1, target_len):

            # Insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # Compute decoder output for one frame
            if frame == 1:
                output, encoder_state = self.decoder(inputs, decoder_hidden)
            else:
                output, encoder_state = self.decoder(inputs, encoder_state)

            outputs[:, frame : frame + 1, :] = output

            # teacher forcing (scheduled sampling)
            teacher_force = random.random() < self.teacher_forcing_ratio

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


class EventEmbedding(nn.Module):
    """Embed one-hot event tensors."""

    def __init__(self, input_dim, output_dim):
        """Initialize single embedding layer."""
        super().__init__()
        self.fc_layer_forward = nn.Linear(input_dim, output_dim)
        self.fc_layer_backward = nn.Linear(input_dim, output_dim)

    def forward(self, event_tensor, encoder_state):
        """Perform forward propagation."""
        # forward direction
        state_forward = torch.cat([encoder_state[0][0], event_tensor], dim=1)
        encoder_state[0][0] = self.fc_layer_forward(state_forward)

        # backward direction
        state_backward = torch.cat([encoder_state[0][1], event_tensor], dim=1)
        encoder_state[0][1] = self.fc_layer_backward(state_backward)

        return encoder_state


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
            [[self.char2id["<BOS>"]] for _ in range(n_batch)],
            device=self.device,
        )
        embedding = self.forward(bos)

        return embedding


def get_model(mapping_dict, cfg):
    """Instantiate models."""
    char2id = mapping_dict.char2id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = {
        "encoder": None,
        "decoder": None,
        "seq2seq": None,
        "event": None,
        "bos": None,
    }

    input_dim = cfg.feature.n_fft // 2 + 1
    output_dim = cfg.feature.n_fft // 2 + 1
    vocab_size = len(char2id)

    model["encoder"] = Encoder(
        dimensions=(vocab_size, cfg.model.hidden_dim, cfg.model.hidden_dim),
        char2id=char2id,
        n_layers=cfg.model.n_layers_enc,
    ).to(device)

    model["decoder"] = Decoder(
        input_dim, output_dim, cfg.model.hidden_dim, n_layers=cfg.model.n_layers_dec
    ).to(device)

    model["event"] = EventEmbedding(
        input_dim=cfg.model.hidden_dim + len(cfg.sound_event),
        output_dim=cfg.model.hidden_dim,
    ).to(device)

    model["seq2seq"] = Seq2Seq(
        {
            "encoder": model["encoder"],
            "decoder": model["decoder"],
            "event": model["event"],
        },
        device,
        criterion=cfg.model.criterion,
        teacher_forcing_ratio=cfg.training.teacher_forcing_ratio,
    )

    model["bos"] = BosEmbedding(vocab_size, output_dim, char2id, device).to(device)

    return model
