import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np


class StupidBotDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, header=0)
        self.questions = self.data["question"]
        self.answers = self.data["answer"]
        self.data_len = len(self.data.index)

        # Unique characters in the database.
        self.unique_characters = set("".join(self.questions + self.answers))
        self.unique_characters_length = len(self.unique_characters)  # 24

        # Map int to character.
        self.int2char = {i: char for i, char in enumerate(self.unique_characters)}
        # Map character to int.
        self.char2int = {char: i for i, char in enumerate(self.unique_characters)}

        # Longer question.
        self.longer_question_length = len(max(self.questions, key=len))  # 24
        # Longer answer.
        self.longer_answer_length = self.longer_question_length
        # self.longer_answer_length = len(max(self.answers, key=len))  # 14

        # Pad strings.
        self.questions = self.questions.str.pad(
            self.longer_question_length, side="right"
        )
        self.answers = self.answers.str.pad(self.longer_answer_length, side="right")

    def __getitem__(self, index):
        x = self.questions[index]
        # Map text to int.
        x = self.text2int(x)
        # One-hot encode x.
        x = self.one_hot_encode(x)
        x = torch.tensor(x).float().cuda()

        y = self.answers[index]
        # Map text to int.
        y = self.text2int(y)
        y = torch.tensor(y).float().cuda()
        return x, y

    def __len__(self):
        return self.data_len

    def text2int(self, text):
        """
            Convert text to an array of integers.
        """
        return [self.char2int[c] for c in text]

    def int2text(self, sequence):
        """
            Convert an array of integers to text.
        """
        return [self.int2char[c] for c in sequence]

    def one_hot_encode(self, sequence):
        """
            Convert an array of integers to a matrix one-hot encoded.
        """
        encoded = np.zeros([len(sequence), self.unique_characters_length], dtype=int)
        for i, character in enumerate(sequence):
            encoded[i][character] = 1
        return encoded

    def one_hot_decode(self, sequence):
        """
            sequence: PyTorch tensor.
        """
        return [np.argmax(x) for x in sequence.numpy()]
