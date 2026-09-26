"""
Shared PyTorch Dataset/collate for audio+text multimodal intent classification.
Reads the flat JSONL files produced by prepare_slurp.py / prepare_fsc.py.

CHANGE FROM PREVIOUS VERSION: added an in-memory cache for decoded+feature-extracted
audio tensors, keyed by file path. Without this, every epoch re-decodes every FLAC
file from disk from scratch -- for a 20-epoch run over 50K files, that's 1 million
redundant disk reads + decodes. With DataLoader(persistent_workers=True) (set in
train.py), each worker process keeps this cache alive across epochs, so only the
FIRST epoch pays the full decode cost; subsequent epochs reuse the cached tensor.

Trade-off: this uses more RAM (roughly: num_examples x audio_tensor_size, held per
worker process). For SLURP at a few seconds of 16kHz audio each, this is a few GB
per worker -- fine on a machine with reasonable RAM, but keep an eye on it if you
increase num_workers a lot.
"""
import json

import librosa
import torch
from torch.utils.data import Dataset


class IntentDataset(Dataset):
    def __init__(self, jsonl_path, label_map, text_tokenizer, audio_feature_extractor,
                 max_text_len=128, max_audio_seconds=10, target_sr=16000, cache_audio=True):
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        self.label_map = label_map
        self.text_tokenizer = text_tokenizer
        self.audio_feature_extractor = audio_feature_extractor
        self.max_text_len = max_text_len
        self.max_audio_seconds = max_audio_seconds
        self.target_sr = target_sr
        self.cache_audio = cache_audio
        self._audio_cache = {} if cache_audio else None

    def __len__(self):
        return len(self.examples)

    def _load_audio_features(self, audio_path):
        if self.cache_audio and audio_path in self._audio_cache:
            return self._audio_cache[audio_path]

        wav, _ = librosa.load(audio_path, sr=self.target_sr, duration=self.max_audio_seconds)
        audio_enc = self.audio_feature_extractor(
            wav, sampling_rate=self.target_sr, return_tensors="pt",
            padding="max_length", truncation=True,
            max_length=self.target_sr * self.max_audio_seconds,
        )
        input_values = audio_enc["input_values"].squeeze(0)

        if self.cache_audio:
            self._audio_cache[audio_path] = input_values
        return input_values

    def __getitem__(self, idx):
        ex = self.examples[idx]
        label_id = self.label_map[ex["label"]]

        text_enc = self.text_tokenizer(
            ex["transcript"], truncation=True, max_length=self.max_text_len,
            padding="max_length", return_tensors="pt",
        )

        input_values = self._load_audio_features(ex["audio_path"])

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "input_values": input_values,
            "label": torch.tensor(label_id, dtype=torch.long),
            "utt_id": ex["utt_id"],
        }


def load_label_map(path):
    with open(path) as f:
        return json.load(f)
