import torch
from torch import nn
from torchtext.vocab import GloVe, vocab
import os
import librosa
from music_label.inference import pred_label
from music_label.labels import labels_adjusted
import torch

glove = GloVe(name="6B", dim=100)
emo_vocab = vocab(glove.stoi)
embedding = nn.Embedding.from_pretrained(glove.vectors)
import random

emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class Database:
    def __init__(self) -> None:
        songs = []
        self.names = []
        for song_path in os.listdir("songlib"):
            print(song_path)
            y, sr = librosa.load(f"songlib/{song_path}")
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96)
            print(melspec.shape)
            if melspec.shape[1] < 2000:
                continue
            beg = 0
            print(int(beg))
            emb = torch.softmax(
                pred_label(torch.tensor([melspec[:, beg : beg + 1366]])).flatten(),
                dim=0,
            )
            print(emb)
            weights = list(emb)
            song_emb = torch.zeros(1, 100)
            for weight, label in zip(weights, labels_adjusted):
                song_emb += (
                    embedding(torch.tensor(emo_vocab.lookup_indices([label]))) * weight
                )
            self.names.append(song_path)
            songs.append(song_emb)

        self.song_emb = torch.vstack(songs)

    def query(self, tensor):
        tensor = tensor.flatten()
        tensor = torch.softmax(tensor, dim=0)
        weights = list(tensor)
        emo_emb = torch.zeros(1, 100)
        for weight, emotion in zip(weights, emotions):
            emo_emb += (
                embedding(torch.tensor(emo_vocab.lookup_indices([emotion]))) * weight
            )

        scores = torch.cosine_similarity(self.song_emb, emo_emb)
        pred = torch.argmax(scores)
        print(self.names[pred])
        return self.names[pred]


db = Database()
db.query(torch.tensor([[-0.8649, -0.5382, -1.5810, 0.3, 3, -0.8571, -0.9041, 0.4438]]))
