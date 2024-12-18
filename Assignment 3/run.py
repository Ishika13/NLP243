import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
import pickle
import csv
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
seq_length = 50
batch_size = 128

def load_glove_embeddings(glove_path, word2idx, embedding_dim=300):
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    embeddings = np.random.randn(len(word2idx), embedding_dim)
    found_words = 0
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                values = line.split()
                word = values[0]
                if word in word2idx:
                    vector = np.asarray(values[1:], dtype='float32')
                    
                    # Verify vector dimension
                    if len(vector) != embedding_dim:
                        print(f"Warning: Line {line_num} has incorrect dimension {len(vector)} (expected {embedding_dim})")
                        continue
                        
                    embeddings[word2idx[word]] = vector
                    found_words += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Loaded {found_words} words from GloVe embeddings.")
    return torch.tensor(embeddings)

def preprocess_data(split='train'):
    ptb = load_dataset('ptb_text_only', split=split, trust_remote_code=True)
    sentences = [sentence['sentence'].split() for sentence in ptb]
    
    vocab = set(word for sentence in sentences for word in sentence)
    vocab.add('<unk>')
    vocab.add('<pad>')
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    tokenized_sentences = [[word2idx.get(word, word2idx['<unk>']) for word in sentence] for sentence in sentences]
    return tokenized_sentences, word2idx, idx2word

def create_dataset(word2idx, tokenized_sentences, seq_length=40):
    data = []
    for sentence in tokenized_sentences:
        if len(sentence) < seq_length + 1:
            sentence = sentence + [word2idx['<pad>']] * (seq_length + 1 - len(sentence))
            
        for i in range(len(sentence) - seq_length):
            input_seq = sentence[i:i + seq_length]
            target_seq = sentence[i + 1:i + seq_length + 1]
            data.append((input_seq, target_seq))
    return data

def prepare_data_loader(data, batch_size=64):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([x[0] for x in data]).to(device),
        torch.tensor([x[1] for x in data]).to(device)
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, num_heads=8, num_layers=6,
                 ff_dim=512, dropout=0.2, pretrained_embeddings=None):
        super().__init__()
        self.padding_idx = word2idx['<pad>']
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.positional_encoding = self._generate_positional_encoding(embedding_dim, 512)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, embedding_dim, max_len):
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(np.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0).to(device)

    def forward(self, x):
        padding_mask = (x == self.padding_idx)
        
        x = self.dropout(self.embedding(x)) + self.positional_encoding[:, :x.size(1), :]
        
        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(device)
        
        output = self.decoder(
            x,
            x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask
        )
        
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=25,
                save_path="decoder_v2_model_epoch_{}.pth"):
    model.train()
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        total_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            targets = targets.contiguous().view(-1)
            
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / total_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path.format(epoch + 1))
            print(f"Model saved to {save_path.format(epoch + 1)}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def evaluate_and_save_perplexity(model, test_sentences, seq_length, word2idx, output_file):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    results = []

    for idx, sentence in enumerate(test_sentences):
        # Ensure the sentence is long enough
        if len(sentence) < seq_length + 1:
            # Pad both input and target sequences
            sentence = sentence + [word2idx['<pad>']] * (seq_length + 1 - len(sentence))
        
        # Take only seq_length tokens for input and target
        input_seq = sentence[:seq_length]
        target_seq = sentence[1:seq_length+1]  # Shift by 1
        
        # Ensure both sequences are of the same length
        assert len(input_seq) == len(target_seq) == seq_length, \
            f"Sequence length mismatch: input={len(input_seq)}, target={len(target_seq)}"
        
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
        target_tensor = torch.tensor(target_seq).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            output = output.reshape(-1, output.size(-1))
            target_tensor = target_tensor.reshape(-1)
            
            loss = criterion(output, target_tensor)
            perplexity = np.exp(loss.item())
        
        results.append((idx, perplexity))

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "ppl"])
        writer.writerows(results)

    print(f"Perplexity results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and save perplexity")
    parser.add_argument("output_file", type=str, help="Output file name")
    args = parser.parse_args()

    # Load and preprocess data
    tokenized_sentences, word2idx, idx2word = preprocess_data('train')

    word2idx_path = "word2idx.pkl"
    with open(word2idx_path, "wb") as f:
        pickle.dump(word2idx, f)
    print(f"word2idx dictionary saved to {word2idx_path}")

    train_data = create_dataset(word2idx, tokenized_sentences, seq_length=seq_length)
    train_loader = prepare_data_loader(train_data, batch_size=batch_size)

    # Load GloVe embeddings
    glove_path = "glove.6B.300d.txt" 
    #glove_path = "/Users/ishikakulkarni/Downloads/glove.6B/glove.6B.300d.txt"
    pretrained_embeddings = load_glove_embeddings(glove_path, word2idx)

    # Initialize model
    vocab_size = len(word2idx)
    model = DecoderOnlyTransformer(
        vocab_size,
        embedding_dim=300,
        num_heads=5,
        num_layers=4,
        ff_dim=128,
        dropout=0.25,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00001,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    # Train the model
    train_model(model, train_loader, nn.CrossEntropyLoss(), torch.optim.AdamW(
        model.parameters(),
        lr=0.00001,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    ), torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    ), epochs=5)

    # Load and evaluate test data
    ptb_test = load_dataset('ptb_text_only', split='test', trust_remote_code=True)
    test_sentences = [sentence['sentence'].split() for sentence in ptb_test]
    test_tokenized = []

    for sentence in test_sentences:
        # Ensure the sentence is long enough for both input and target (seq_length + 1)
        if len(sentence) < seq_length + 1:
            sentence = sentence + ['<pad>'] * (seq_length + 1 - len(sentence))
        # Take only the first seq_length + 1 tokens if sentence is longer
        sentence = sentence[:seq_length + 1]
        temp = [word2idx.get(word, word2idx['<unk>']) for word in sentence]
        test_tokenized.append(temp)

    # Evaluate and save results
    model.eval()
    evaluate_and_save_perplexity(model, test_tokenized, seq_length=seq_length, word2idx=word2idx, output_file=args.output_file)