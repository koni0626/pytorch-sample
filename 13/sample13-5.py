import torch
import torch.nn as nn
import torch.optim as optim

# データの準備
text = """
hello world good morning see you hello world good night
it's always a pleasure to meet you in the beautiful morning
how has your day been so far
night is calm and serene
sunshine brings hope and light to our world
every morning we embark on a new journey
dreams are the essence of our night
"""
words = text.split()
vocab = set(words)
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(vocab)

# パラメータの設定
input_size = vocab_size
hidden_size = 50
output_size = vocab_size
num_layers = 1
seq_length = 5

# モデルの定義
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(num_layers, batch_size, self.hidden_size),
                torch.zeros(num_layers, batch_size, self.hidden_size))

# 学習
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    total_loss = 0
    hidden = model.init_hidden(1)
    for i in range(0, len(words) - seq_length, seq_length):
        input_sequence = torch.tensor([word_to_index[word] for word in words[i:i+seq_length]], dtype=torch.long).unsqueeze(0)
        target_sequence = torch.tensor([word_to_index[word] for word in words[i+1:i+seq_length+1]], dtype=torch.long).view(-1)

        optimizer.zero_grad()
        output, hidden = model(input_sequence, hidden)
        output = output.view(-1, vocab_size)
        loss = criterion(output, target_sequence)
        loss.backward()
        optimizer.step()

        hidden = (hidden[0].detach(), hidden[1].detach())
        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss}")

# 予測
def predict_next_word(word_sequence):
    with torch.no_grad():
        input_sequence = torch.tensor([word_to_index[word] for word in word_sequence], dtype=torch.long).unsqueeze(0)
        hidden = model.init_hidden(1)
        output, _ = model(input_sequence, hidden)
        predicted_index = torch.argmax(output[0][-1]).item()
        return index_to_word[predicted_index]

word_sequence = ["good", "morning", "see", "you", "hello"]
predicted_word = predict_next_word(word_sequence)
print(f"After the sequence '{' '.join(word_sequence)}', the predicted next word is '{predicted_word}'")
