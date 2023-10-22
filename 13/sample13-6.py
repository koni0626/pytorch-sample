import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from itertools import chain
from janome.tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, filepath, seq_length):
        self.seq_length = seq_length
        tokenizer = Tokenizer()

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        tokens = list(chain(*[tokenizer.tokenize(line.strip(), wakati=True) for line in lines]))
        vocabulary = set(tokens)
        self.vocab_size = len(vocabulary)
        self.word_to_index = {word: idx for idx, (word, _) in enumerate(Counter(tokens).items())}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        self.data = [self.word_to_index[word] for word in tokens]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        tokens = list(self.data[idx: idx + self.seq_length + 1])
        input_sequence = tokens[:self.seq_length]
        target_sequence = tokens[1:]
        return torch.tensor(input_sequence), torch.tensor(target_sequence)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

    def predict_next_word(self, input_string, tokenizer, word_to_index, index_to_word):
        self.eval()
        input_sequence = tokenizer.tokenize(input_string, wakati=True)
        input_sequence = [word_to_index.get(word, 0) for word in input_sequence]
        input_tensor = torch.tensor(input_sequence).unsqueeze(0).cuda()
        with torch.no_grad():
            output = self(input_tensor)
            predicted_output_index = torch.argmax(output[0][-1]).item()
        return index_to_word[predicted_output_index]

filepath = 'data.txt'
seq_length = 5
dataset = TextDataset(filepath, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

vocab_size = dataset.vocab_size
embedding_dim = 256
hidden_dim = 512
output_dim = vocab_size

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
model = model.cuda()  # Move model to GPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 5
for epoch in range(epochs):
    for input_sequences, target_sequences in dataloader:
        optimizer.zero_grad()

        input_sequences, target_sequences = input_sequences.cuda(), target_sequences.cuda()  # Move to GPU
        outputs = model(input_sequences)
        loss = criterion(outputs.view(-1, vocab_size), target_sequences.view(-1))
        loss.backward()
        optimizer.step()

print("Training Finished!")
# 予測テスト
# 予測テスト
tokenizer = Tokenizer()
input_text = "アメリカのポー"

for i in range(200):
    predicted_word = model.predict_next_word(input_text, tokenizer, dataset.word_to_index, dataset.index_to_word)
    #print("Predicted word:", predicted_word)
    input_text += predicted_word
    print(input_text)
