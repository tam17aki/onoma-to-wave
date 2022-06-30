# -*- coding: utf-8 -*-
"""Trainer class definition for Onoma-to-Wave model (conditioned on sound events).

Copyright (C) 2022 by Akira TAMAMORI

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
import os

import torch
import torch.nn.functional as F
from progressbar import progressbar as prg

from optimizer import get_scheduler

torch.backends.cudnn.benchmark = True  # a magic


class Trainer:
    """Trainer Class."""

    def __init__(self, model, optimizer, cfg):
        """Initialize class."""
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _training_step(self, batch):
        """Perform a training step."""
        onomatopoeia, specs, frame_length, event_label = batch

        onomatopoeia = onomatopoeia.to(self.device)
        specs = specs.to(self.device).float()
        event_label = event_label.to(self.device).long()
        event_label = F.one_hot(event_label, num_classes=len(self.cfg.sound_event))

        # embedding of <BOS>
        bos_embedding = self.model["bos"].bos_embedding(self.cfg.training.n_batch)

        # attach <BOS> to the front frame
        target = torch.cat([bos_embedding, specs], dim=1)

        loss = self.model["seq2seq"].get_loss(
            source=onomatopoeia,
            target=target,
            frame_lengths=frame_length,
            event_label=event_label,
        )

        return loss

    def fit(self, dataloader):
        """Perform model training."""

        if self.cfg.training.use_scheduler:
            scheduler = get_scheduler(self.optimizer, self.cfg)

        self.model["seq2seq"].train()
        self.model["event"].train()

        for epoch in prg(
            range(1, self.cfg.training.n_epoch + 1),
            prefix="Model training: ",
            suffix="\n",
        ):
            epoch_loss = 0
            for data in dataloader:
                self.optimizer["seq2seq"].zero_grad()
                self.optimizer["event"].zero_grad()
                loss = self._training_step(data)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer["seq2seq"].step()
                self.optimizer["event"].step()

            epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch}: loss = {epoch_loss:.6f}")

            if self.cfg.training.use_scheduler:
                scheduler["seq2seq"].step()

    def save(self):
        """Save model parameters."""
        model_dir = os.path.join(
            self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.model_dir
        )
        os.makedirs(model_dir, exist_ok=True)

        n_epoch = self.cfg.training.n_epoch
        n_batch = self.cfg.training.n_batch
        learning_rate = self.cfg.training.learning_rate
        prefix = self.cfg.training.model_prefix
        filename = os.path.join(
            model_dir, f"{prefix}_epoch{n_epoch}_batch{n_batch}_lr{learning_rate}.pt"
        )
        torch.save(
            {
                "seq2seq": self.model["seq2seq"].state_dict(),
                "event": self.model["event"].state_dict(),
            },
            filename,
        )
