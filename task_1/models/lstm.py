import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM-based recommender system
    """

    def __init__(self,num_items,embedding_dim=32,hidden_dim=64):
        super().__init__()
        self.embedding=nn.Embedding(num_items,embedding_dim)

        self.lstm=nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc=nn.Linear(hidden_dim,num_items)

    def forward(self,seq):
        x=self.embedding(seq)

        _,(hidden,_)=self.lstm(x)
        x=hidden[-1]
        x=self.fc(x)
        return x