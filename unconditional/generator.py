# -*- coding: utf-8 -*-
"""Generator class definition for Onoma-to-Wave model (NOT conditioned on sound events).

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
import soundfile as sf
import torch
from progressbar import progressbar as prg


class Generator:
    """Generator Class for generating audio."""

    def __init__(self, model, mapping_dict, cfg):
        """Initialize class."""
        self.model = model
        self.char2id = mapping_dict.char2id
        self.id2char = mapping_dict.id2char
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(
            os.path.join(
                cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.stats_dir, cfg.training.scaler_file
            )
        )

    def _get_nframes(self, wavfile):
        """Get the shape of the extracted spectrogram."""
        audio, _ = librosa.load(wavfile, sr=self.cfg.feature.sample_rate, mono=True)

        frames = librosa.util.frame(
            audio,
            frame_length=self.cfg.feature.win_length,
            hop_length=self.cfg.feature.hop_length,
        )
        return frames.shape[1]

    def _inference_step(self, batch):
        """Perform a inference step."""
        onomatopoeia, wavfiles = batch

        onomatopoeia = onomatopoeia.to(self.device)
        n_frame = self._get_nframes(wavfiles[0])
        bos_embedding = self.model["bos"].bos_embedding(n_batch=1)

        # Generate standardized spectrogram
        outputs = self.model["seq2seq"].forward(
            onomatopoeia, bos_embedding, n_frame=n_frame
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

    def _get_path(self, batch, gen_dir):
        """Get path for generated audio to be saved."""
        onomatopoeia, wavfiles = batch

        wav_path = wavfiles[0].replace(
            os.path.join(self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.wav_dir), gen_dir
        )
        dirname = os.path.dirname(wav_path)
        os.makedirs(dirname, exist_ok=True)

        basename = os.path.splitext(os.path.basename(wav_path))[0]
        event_name = wavfiles[0].split("/")[-3]

        # phoneme sequence in numerical value
        onoma_idlist = onomatopoeia[0].detach().numpy().copy()

        # phoneme sequence in character
        translation = "".join(list(map(lambda id: self.id2char[id], onoma_idlist)))

        basename = f"{event_name}_" + basename + f"_{translation}.wav"

        return os.path.join(dirname, basename)

    def generate(self, dataloader):
        """Audio generation with trained models."""
        gen_dir = os.path.join(self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.gen_dir)
        os.makedirs(gen_dir, exist_ok=True)

        self.model["seq2seq"].eval()
        self.model["bos"].eval()

        for data in prg(dataloader, prefix="Audio generation by trained models "):
            with torch.no_grad():
                audio = self._inference_step(data)
                wavpath = self._get_path(data, gen_dir)
                sf.write(wavpath, audio, self.cfg.feature.sample_rate)

    def load_model_params(self):
        """Load model parameters for inference."""
        n_epoch = self.cfg.training.n_epoch
        n_batch = self.cfg.training.n_batch
        learning_rate = self.cfg.training.learning_rate
        prefix = self.cfg.training.model_prefix
        model_dir = os.path.join(
            self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.model_dir
        )
        model_file = os.path.join(
            model_dir, f"{prefix}_epoch{n_epoch}_batch{n_batch}_lr{learning_rate}.pt"
        )
        checkpoint = torch.load(model_file)
        self.model["seq2seq"].load_state_dict(checkpoint["seq2seq"])
