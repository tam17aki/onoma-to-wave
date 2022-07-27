# -*- coding: utf-8 -*-
"""Demonstration script for audio generation using pretrained model.

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

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import DictConfig

from mapping_dict import MappingDict
from models import get_model


class GeneratorDemo:
    """GeneratorDemo class for demonstration of audio generation."""

    def __init__(self, cfg):
        """Initialize class."""
        mapping_dict = MappingDict(cfg)
        mapping_dict.load()

        self.model = get_model(mapping_dict, cfg)
        self.char2id = mapping_dict.char2id
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(
            os.path.join(
                cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.stats_dir, cfg.training.scaler_file
            )
        )

    def _inference_step(self, onomatopoeia, sound_event, n_frame=32):
        """Perform a inference step."""
        onomatopoeia = onomatopoeia.to(self.device)
        event_label = sound_event.to(self.device).long()
        event_label = F.one_hot(event_label, num_classes=len(self.cfg.sound_event))

        bos_embedding = self.model["bos"].get_embedding(n_batch=1)
        outputs = self.model["seq2seq"].forward(
            onomatopoeia, bos_embedding, event_label, n_frame=n_frame
        )

        # Move data on torch tensor to numpy array num
        outputs = outputs.to("cpu").detach().numpy().copy()
        outputs = outputs.squeeze(0)

        # Scale back to the original
        outputs = self.scaler.inverse_transform(outputs).T

        # log-scale amplitude -> linear-scale amplitude
        outputs = librosa.db_to_amplitude(outputs)

        # Convert spectrogram to audio (addition of phase spectrogram)
        outputs = librosa.griffinlim(
            outputs,
            n_fft=self.cfg.feature.n_fft,
            hop_length=self.cfg.feature.hop_length,
            win_length=self.cfg.feature.win_length,
            n_iter=self.cfg.feature.n_iter,
        )
        audio = librosa.util.normalize(outputs)

        return audio

    def _get_onoma_tensor(self):
        """Get tensor of onomatopoeia."""
        onomatopoeia = self.cfg.demo.onomatopoeia
        chars = onomatopoeia.split()
        char_ids = [self.char2id[c] for c in chars]
        return torch.from_numpy(np.array(char_ids)).unsqueeze(0)

    def _get_event_tensor(self):
        """Get tensor of sound event label."""
        sound_event = torch.tensor(
            self.cfg.sound_event.index(self.cfg.demo.sound_event)
        )
        return sound_event.unsqueeze(0)

    def _get_wavpath(self, gen_dir):
        """Get path for generated audio to be saved."""
        onomatopoeia = self.cfg.demo.onomatopoeia
        chars = onomatopoeia.split()
        onoma_seq = "".join(chars)
        event = self.cfg.demo.sound_event
        wavpath = os.path.join(gen_dir, self.cfg.demo.basename)
        wavpath = wavpath + f"{onoma_seq}_{event}.wav"
        return wavpath

    def generate(self):
        """Generate audio with a trained model."""
        gen_dir = os.path.join(self.cfg.RWCP_SSD.root_dir, self.cfg.demo.gen_dir)
        os.makedirs(gen_dir, exist_ok=True)

        self.model["seq2seq"].eval()
        self.model["bos"].eval()

        onomatopoeia = self._get_onoma_tensor()
        event_label = self._get_event_tensor()
        wavpath = self._get_wavpath(gen_dir)

        with torch.no_grad():
            audio = self._inference_step(
                onomatopoeia, event_label, self.cfg.demo.n_frame
            )
            sf.write(wavpath, audio, self.cfg.feature.sample_rate)

    def load_model_params(self):
        """Load model parameters for inference."""
        checkpoint = torch.load(self.cfg.demo.pretrained_model)
        self.model["seq2seq"].load_state_dict(checkpoint)


def main(cfg: DictConfig):
    """Perform inference."""
    generator = GeneratorDemo(cfg)
    generator.load_model_params()
    generator.generate()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
