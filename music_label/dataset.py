from jamendo.scripts.commons import read_file
import numpy as np
import random
import torch
from labels import labels as moods


def make_splits():
    tracks, tags, _ = read_file("jamendo/data/autotagging_moodtheme.tsv")

    """
    tags
    {
        "mood/theme": {
            "dream" : [id_1, id_2, ...]
            "sad": [id_1, id_2, ...]
        }
    }
    """

    """
    tracks
    {
        id_1: {...}
        id_2: {...}
    }
    """

    songs = set()

    for mood in moods:
        for song in tags["mood/theme"][mood]:
            songs.add(song)

    dataset = []

    for song in list(songs):
        with open(f"melspecs/{str(song)[-2:]}/{str(song)}.npy", "rb") as file:
            data = torch.tensor(np.load(file))
            if data.shape[1] < 2000:
                continue
            beg = random.randint(300, data.shape[1] - 1700)
            tgt = torch.zeros(len(moods))
            for label in tracks[song]["mood/theme"]:
                if label in moods:
                    tgt[moods.index(label)] = 1

            dataset.append((data[:, beg : beg + 1366], tgt))

    random.shuffle(dataset)
    test = dataset[:1000]
    train = dataset[1000:]

    test_src, test_tgt = list(zip(*test))
    test_src = torch.stack(test_src)
    test_tgt = torch.vstack(test_tgt)
    print(test_src.shape)
    print(test_tgt.shape)

    train_src, train_tgt = list(zip(*train))
    train_src = torch.stack(train_src)
    train_tgt = torch.vstack(train_tgt)
    print(train_src.shape)
    print(train_tgt.shape)

    torch.save(test_src, "dataset/test_src.pt")
    torch.save(test_tgt, "dataset/test_tgt.pt")
    torch.save(train_src, "dataset/train_src.pt")
    torch.save(train_tgt, "dataset/train_tgt.pt")


if __name__ == "__main__":
    make_splits()
