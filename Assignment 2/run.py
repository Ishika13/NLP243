import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
from sklearn.model_selection import train_test_split
import zipfile
import requests
import matplotlib.pyplot as plt

# Glove embeddings (you can keep this as is or modify it to handle paths if necessary)
def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Define VocabularyEmbedding class
class VocabularyEmbedding:
    def __init__(self, glove_embeddings, embedding_dim=300):
        self.embedding_dim = embedding_dim
        self.embeddings = glove_embeddings
        self.pad_index = 0
        self.unk_index = 1
        self.word2idx = {'<pad>': self.pad_index, '<unk>': self.unk_index}
        self.idx2word = {self.pad_index: '<pad>', self.unk_index: '<unk>'}
        self.vectors = [np.zeros(embedding_dim), np.random.uniform(-1, 1, embedding_dim)]

        for word, vector in glove_embeddings.items():
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.vectors.append(vector)

        self.vectors = torch.FloatTensor(self.vectors)

    def encode(self, sentence):
        return [self.word2idx.get(word, self.unk_index) for word in sentence.split()]

    def create_padded_tensor(self, sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded = torch.full((len(sequences), max_len), self.pad_index, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded, lengths

# Define IOBTagSequencer class
class IOBTagSequencer:
    def __init__(self, tag_corpus):
        self.word2idx = {'O': 0}
        self.idx2word = {0: 'O'}
        for tags in tag_corpus:
            for tag in tags.split():
                if tag not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[tag] = idx
                    self.idx2word[idx] = tag

    def encode(self, tags):
        return [self.word2idx[tag] for tag in tags.split()]

    def create_padded_tensor(self, sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded = torch.full((len(sequences), max_len), 0, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded, lengths

# BiLSTM Model Definition
class BiLSTMModel(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, vocab_size, embeddings, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)  
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2) 
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        output = output.view(-1, output.size(2)) 
        output = self.batch_norm(output)  
        output = output.view(x.size(0), x.size(1), -1)  
        output = self.dropout(output)
        return self.fc(output)

# Dataset class for data loading
class Dataset(Dataset):
    def __init__(self, data, text_sequencer, tag_sequencer):
        self.data = data
        self.text_sequencer = text_sequencer
        self.tag_sequencer = tag_sequencer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, tags = self.data[index]
        x = self.text_sequencer.encode(text)
        y = self.tag_sequencer.encode(tags)
        return x, y

# Custom collate function for padding
def collate_fn(batch, text_sequencer, tag_sequencer):
    texts, tags = zip(*batch)
    padded_texts, text_lengths = text_sequencer.create_padded_tensor(texts)
    padded_tags, _ = tag_sequencer.create_padded_tensor(tags)
    return padded_texts, padded_tags, text_lengths


# Evaluation function
def evaluate_model(model, data_loader, device, tag_sequencer, loss_function):
    model.eval()
    total_loss = 0
    true_tags = []
    pred_tags = []

    with torch.no_grad():
        for x_batch, y_batch, lengths in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch, lengths)

            logits_flat = logits.view(-1, logits.shape[-1])
            y_batch_flat = y_batch.view(-1)
            loss = loss_function(logits_flat, y_batch_flat)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=2)

            for i in range(len(y_batch)):
                true_tag_seq = [tag_sequencer.idx2word[idx.item()] for idx in y_batch[i][:lengths[i]]]
                pred_tag_seq = [tag_sequencer.idx2word[idx.item()] for idx in predictions[i][:lengths[i]]]

                true_tags.append(true_tag_seq)
                pred_tags.append(pred_tag_seq)

        print("Classification Report:")
        print(classification_report(true_tags, pred_tags, scheme=IOB2))

        precision = precision_score(true_tags, pred_tags)
        recall = recall_score(true_tags, pred_tags)
        f1 = f1_score(true_tags, pred_tags)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        val_loss = total_loss / len(data_loader)

        return val_loss, precision, recall, f1

# Main training and evaluation function
def train_and_evaluate(train_csv_path, test_csv_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load GloVe embeddings
    glove_path = 'glove.6B.300d.txt'
    if not os.path.exists(glove_path):
        print("Downloading GloVe embeddings...")
        url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        response = requests.get(url)
        with open('glove.6B.zip', 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall()
    
    glove_embeddings = load_glove_embeddings(glove_path)

    # Setup the text and tag sequencer
    text_sequencer = VocabularyEmbedding(glove_embeddings)
    df_train = pd.read_csv(train_csv_path)
    df_train['IOB Slot tags'] = df_train['IOB Slot tags'].replace(np.nan, "O")
    item_list_train = [(row['utterances'], row['IOB Slot tags']) for _, row in df_train.iterrows()]
    train_data, validation_data = train_test_split(item_list_train, test_size=0.05, random_state=42)

    tag_sequencer = IOBTagSequencer([tags for _, tags in train_data])

    train_dataset = Dataset(train_data, text_sequencer, tag_sequencer)
    val_dataset = Dataset(validation_data, text_sequencer, tag_sequencer)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          collate_fn=lambda batch: collate_fn(batch, text_sequencer, tag_sequencer))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        collate_fn=lambda batch: collate_fn(batch, text_sequencer, tag_sequencer))

    output_size = len(tag_sequencer.idx2word)
    hidden_dim = 256
    model = BiLSTMModel(output_size, 300, hidden_dim, len(text_sequencer.word2idx), text_sequencer.vectors).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_function = nn.CrossEntropyLoss()

    # Training loop
    epochs = 2
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch, lengths)

            logits_flat = logits.view(-1, logits.shape[-1])
            y_batch_flat = y_batch.view(-1)

            loss = loss_function(logits_flat, y_batch_flat)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f"Train Loss for Epoch {epoch+1}: {train_losses[-1]:.4f}")

        val_loss, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, device, tag_sequencer, loss_function
        )
        val_losses.append(val_loss)
        print(f"Validation Loss for Epoch {epoch+1}: {val_loss:.4f}")
        print(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    # Plot training and validation loss
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save predictions to output file
    df_test = pd.read_csv(test_csv_path)
    test_data = [(row['utterances'], "O") for _, row in df_test.iterrows()]
    test_dataset = Dataset(test_data, text_sequencer, tag_sequencer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, text_sequencer, tag_sequencer))


    model.eval()
    predictions = []

    with torch.no_grad():
        for x_batch, _, lengths in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch, lengths)

            predictions_batch = torch.argmax(logits, dim=2)

            for i in range(len(predictions_batch)):
                pred_tag_seq = [tag_sequencer.idx2word[idx.item()] for idx in predictions_batch[i][:lengths[i]]]
                predictions.append(pred_tag_seq)

    df_test['predicted_tags'] = [' '.join(tags) for tags in predictions]
    df_test[['utterances', 'predicted_tags']].to_csv(output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BiLSTM Model for Sequence Labeling")
    parser.add_argument('train_data', type=str, help="Path to training data CSV file")
    parser.add_argument('test_data', type=str, help="Path to test data CSV file")
    parser.add_argument('output', type=str, help="Path to save the predictions")
    args = parser.parse_args()

    train_and_evaluate(args.train_data, args.test_data, args.output)
