import torch
import torch.nn as nn
import torch.optim as optim
import string
import random

# Dados de exemplo (corpus em português)
corpus = "Exemplo de texto em português para treinar uma rede neural, quanto melhor for a qualidadedo input linguístico melhor será a qualidadedo output."

# Pré-processamento: mapear caracteres para índices
charset = string.printable
char_to_index = {char: i for i, char in enumerate(charset)}
index_to_char = {i: char for i, char in enumerate(charset)}

# Função para gerar pares de entrada e saída
def create_data(corpus, sequence_length=10):
    data = []
    for i in range(len(corpus) - sequence_length):
        input_seq = corpus[i:i+sequence_length]
        target_char = corpus[i+sequence_length]
        data.append((input_seq, target_char))
    return data

# Converter dados para índices
data = create_data(corpus)
data_indices = [([char_to_index[char] for char in seq], char_to_index[target]) for seq, target in data]

# Definir a arquitetura da RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.rnn(x.view(1, 1, -1), hidden)
        x = self.fc(x.view(1, -1))
        return x, hidden

# Parâmetros da rede
input_size = len(charset)
hidden_size = 64
output_size = len(charset)

# Instanciar a rede
rnn = RNN(input_size, hidden_size, output_size)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.01)

# Treinamento da rede
num_epochs = 1000
for epoch in range(num_epochs):
    random.shuffle(data_indices)
    total_loss = 0
    hidden = torch.zeros(1, 1, hidden_size)

    for sequence, target in data_indices:
        rnn.zero_grad()
        loss = 0

        for char_index in sequence:
            char_tensor = torch.tensor([char_index], dtype=torch.long)
            output, hidden = rnn(char_tensor, hidden)
            loss += criterion(output, torch.tensor([target], dtype=torch.long))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss/len(data_indices)}')

# Função para gerar texto
def generate_text(starting_seq, length=100):
    with torch.no_grad():
        rnn.eval()
        hidden = torch.zeros(1, 1, hidden_size)
        generated_text = starting_seq

        for _ in range(length):
            input_seq = torch.tensor([char_to_index[char] for char in starting_seq[-sequence_length:]], dtype=torch.long)
            output, hidden = rnn(input_seq, hidden)
            softmax_output = torch.nn.functional.softmax(output[0], dim=0)
            predicted_index = torch.multinomial(softmax_output, 1).item()
            predicted_char = index_to_char[predicted_index]
            generated_text += predicted_char
            starting_seq += predicted_char

        rnn.train()

        return generated_text

# Gerar texto com a rede treinada
generated_text = generate_text("Exemplo de texto", length=200)
print(generated_text)
