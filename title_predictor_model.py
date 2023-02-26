import torch
import torch.nn as nn
import torch.nn.functional as F

# Title predictor model.
class TitlePredictor(nn.Module):
    def __init__(
        self,
        vocabSize,
        embeddingSize,
        hiddenSize,
        numLayers,
        dropout,
        embedding: nn.Embedding = None
    ):
        super(TitlePredictor, self).__init__()
        # Save parameters.
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.dropoutPr = dropout
        
        if embedding:
            # Check vocab size.
            assert embedding.num_embeddings == vocabSize
            # Check embedding size.
            assert embedding.embedding_dim == embeddingSize
            
            self.embedding = embedding
            print("Using pretrained embedding.")
        else:
            self.embedding = nn.Embedding(vocabSize, embeddingSize)
            print("Using random embedding.")
        self.lstm = nn.LSTM(embeddingSize, hiddenSize, numLayers, dropout=dropout)
        self.fc1 = nn.Linear(hiddenSize*3, hiddenSize*2)
        self.fc2 = nn.Linear(hiddenSize*2, hiddenSize)
        self.linear = nn.Linear(hiddenSize, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [sentence length]
        embedded = self.embedding(x)
        # embedded: [sentence length, embedding size]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [sentence length, hidden size]
        # hidden: [num layers, hidden size]
        # cell: [num layers, hidden size]
        # Combine the last output and the last hidden state.
        hidden = torch.cat((outputs[-1], hidden[-1], cell[-1]), dim=1)
        x = self.fc1(hidden)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        prediction = self.linear(x)
        # prediction: [batch size, 1]
        return prediction