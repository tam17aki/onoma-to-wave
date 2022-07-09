# -*- coding: utf-8 -*-
"""Dataset definition for Onoma-to-Wave model (conditioned on sound events).

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
import functools
import glob
import os
import random

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
from progressbar import progressbar as prg
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class OnomatopoeiaDataset(Dataset):
    """Dataset class for Onoma-to-wave."""

    def __init__(self, char2id, cfg, is_train=True):
        """Initialize class."""
        super().__init__()
        if is_train:
            csv_dir = os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.onoma_traindir)
        else:
            csv_dir = os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.onoma_testdir)

        self.csv_files = [
            os.path.abspath(csv_file)
            for csv_file in glob.glob(csv_dir + "/**/*.csv", recursive=True)
            if any(event_name in csv_file for event_name in cfg.sound_event)
        ]
        self.wav_files = []
        self.items = {
            "onoma": [],  # stores phonetic sequences of onomatopoeia
            "specs": [],  # stores spectrograms
            "wavfiles": [],  # stores path names of audio files
            "event_labels": [],  # stores labels of sound event
        }
        self.stats = {"mean": None, "std": None}
        self.cfg = cfg
        self.is_train = is_train

        spec_list = []
        wav_path_list = []

        if is_train:
            wav_dir = os.path.join(cfg.RWCP_SSD.root_dir, cfg.RWCP_SSD.wav_dir, "train")
            self.wav_files = [
                os.path.abspath(wavfile)
                for wavfile in glob.glob(wav_dir + "/**/*.wav", recursive=True)
                if any(event_name in wavfile for event_name in cfg.sound_event)
            ]

            # Extract spectrograms from audio
            spec_list, wav_path_list = self._extract_spec()

            # Perform standardization (z-norm)
            spec_list = self._compute_standardization(spec_list)

        # Create a list from the acquired acoustic events
        events = self._aggregate_acoustic_events()

        # Get a list of unique acoustic events
        self.event_list = np.unique(np.array(events)).tolist()

        # Pack spectrograms and other necessary items
        self._pack_items(char2id, spec_list, wav_path_list)

    def __len__(self):
        """Return dataset size."""
        return len(self.items["onoma"])

    def __getitem__(self, idx):
        """Fetch items."""
        if self.is_train:
            return (
                self.items["onoma"][idx],
                self.items["specs"][idx],
                self.items["event_labels"][idx],
            )
        return (
            self.items["onoma"][idx],
            self.items["wavfiles"][idx],
            self.items["event_labels"][idx],
        )

    def _extract_spec(self):
        """Extract spectrograms from audio."""
        spec_list = []
        wav_path_list = []

        for wav_file in prg(self.wav_files, prefix="Extract spectrograms: "):
            audio, _ = librosa.load(
                wav_file, sr=self.cfg.feature.sample_rate, mono=True
            )
            audio = librosa.util.normalize(audio)

            # remove silence
            audio, _ = librosa.effects.trim(
                audio,
                top_db=self.cfg.feature.top_db,
                frame_length=self.cfg.feature.win_length,
                hop_length=self.cfg.feature.hop_length,
            )

            # [n_fbin, n_frames]
            spec = librosa.stft(
                y=audio,
                n_fft=self.cfg.feature.n_fft,
                win_length=self.cfg.feature.win_length,
                hop_length=self.cfg.feature.hop_length,
            )
            spec = librosa.amplitude_to_db(np.abs(spec))  # log-scaled amplitude
            spec_list.append(np.transpose(spec))  # [n_frames, n_fbin]

            spec_tmp = wav_file.split("/")
            wav_path_list.append("/".join(spec_tmp[-7:]))

        return spec_list, wav_path_list

    def _compute_standardization(self, spec_list, axis=0):
        """Compute mean and standard dev."""
        # Concatenate along time-axis
        spec = np.concatenate(spec_list, axis=axis)

        scaler = StandardScaler()
        scaler.fit(spec)

        for i, data in enumerate(spec_list):
            spec_list[i] = scaler.transform(data)

        # Save scaler
        stats_dir = os.path.join(
            self.cfg.RWCP_SSD.root_dir, self.cfg.RWCP_SSD.stats_dir
        )
        os.makedirs(stats_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(stats_dir, self.cfg.training.scaler_file))

        return spec_list

    def _aggregate_acoustic_events(self):
        """Aggregate acoustic events."""
        events = []
        for word in prg(self.csv_files, prefix="Aggregate acoustic events: "):
            dataframe = pd.read_csv(word, header=None)
            for i in range(len(dataframe)):
                events.append(dataframe.iat[i, 3])

        return events

    def _pack_items(self, char2id, spec_list, wav_path_list):
        """Pack spectrograms and other necessary items."""
        for word in prg(self.csv_files, prefix="Pack necessary items: "):
            tmp_df = pd.read_csv(word, header=None)

            if self.is_train:
                # In training, 15 onomatopoeia are targeted.
                # -> take 15 lines at random.
                dataframe = tmp_df.sample(
                    n=self.cfg.training.n_onomas, random_state=0, axis=0
                )
            else:
                dataframe = tmp_df[0 : self.cfg.inference.n_onomas]

            for i in range(len(dataframe)):

                # Converts onomatopoeic phoneme sequences to numeric sequences.
                # phoneme seq. -> list of phoneme characters
                char = dataframe.iat[i, 1].split()
                # ex. 'p o N q' -> ['p', 'o', 'N', 'q']

                # Converts each phonetic character to a numerical value
                char_ids = [char2id[c] for c in char]

                self.items["onoma"].append(np.array(char_ids))

                if self.is_train:
                    # "/".join(df.iat[i, 2].split("/")[-4:])
                    # = 'a1/cherry1/16khz/005.wav'
                    # Pull up the spectrogram corresponding
                    # to the wav by the index function of the list
                    wav_path = os.path.join(
                        self.cfg.RWCP_SSD.wav_dir,
                        "train",
                        "/".join(dataframe.iat[i, 2].split("/")[-4:]),
                    )
                    self.items["specs"].append(spec_list[wav_path_list.index(wav_path)])
                else:
                    self.items["wavfiles"].append(dataframe.iat[i, 2])

                # Get the index of acoustic events corresponding to wav
                self.items["event_labels"].append(
                    self.event_list.index(dataframe.iat[i, 3])
                )


def _padding_spec(spec, max_frame_len, padding_value=0.0):
    """Pad zeros to spectrogram."""
    spec_pad = np.pad(
        spec,
        [
            (0, max_frame_len - spec.shape[0]),  # axis=0: time axis
            (0, 0),  # axix=1: frequency axis
        ],
        "constant",
        constant_values=padding_value,
    )

    return spec_pad


def _padding_onoma(onoma, max_len, padding_value=1):
    """Pad constants to phoneme sequence."""
    onoma_pad = np.pad(
        onoma,
        [
            (0, max_len - onoma.shape[0]),
        ],
        "constant",
        constant_values=padding_value,
    )

    return onoma_pad


def _collate_fn_onoma(batch, padvalue_onoma=1, padvalue_spec=0.0, is_train=True):
    """Collate function."""
    onomas, _, event_labels = list(zip(*batch))
    seq_lengths = [len(x) for x in onomas]
    max_len = max(seq_lengths)
    onoma_batch = torch.stack(
        [
            torch.from_numpy(_padding_onoma(onoma, max_len, padvalue_onoma))
            for onoma in onomas
        ]
    )
    event_batch = torch.tensor(event_labels)

    if is_train:
        _, specs, _ = list(zip(*batch))
        frame_lengths = [x.shape[0] for x in specs]
        max_len = max(frame_lengths)
        spec_batch = torch.stack(
            [
                torch.from_numpy(_padding_spec(spec, max_len, padvalue_spec))
                for spec in specs
            ]
        )
        return onoma_batch, spec_batch, frame_lengths, event_batch

    _, wavfiles, _ = list(zip(*batch))
    return onoma_batch, wavfiles, event_batch


def _worker_init_fn(worker_id):
    """Initialize worker functions."""
    random.seed(worker_id)


def get_dataloader(mapping_dict, cfg, is_train=True):
    """Return dataloader from dataset."""
    char2id = mapping_dict.char2id
    dataset = OnomatopoeiaDataset(char2id, cfg, is_train)
    padvalue_onoma = char2id[" "]
    padvalue_spec = cfg.training.padvalue_spec

    if is_train:
        collate_fn_onoma = functools.partial(
            _collate_fn_onoma,
            padvalue_onoma=padvalue_onoma,  # padding for onomatopoeia
            padvalue_spec=padvalue_spec,  # padding for spectrogram
            is_train=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.n_batch,
            shuffle=True,
            drop_last=True,
            num_workers=2,  # for faster computation
            pin_memory=True,  # for faster computation
            worker_init_fn=_worker_init_fn,
            collate_fn=collate_fn_onoma,
        )
    else:
        collate_fn_onoma = functools.partial(
            _collate_fn_onoma, padvalue_onoma=padvalue_onoma, is_train=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn_onoma,
        )

    return dataloader
