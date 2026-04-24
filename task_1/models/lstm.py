import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM-based recommender system
    """

    def __init__(self,num_items,embedding_dim=32,hidden_dim=64):
        super().__init__()
        self.item_emb=nn.Embedding(num_items+1,embedding_dim,padding_idx=0)
        
        self.dow_emb=nn.Embedding(8,embedding_dim,padding_idx=0)
        self.hour_emb=nn.Embedding(25,embedding_dim,padding_idx=0)

        self.merge = nn.Linear(embedding_dim*3,embedding_dim)

        self.lstm=nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc=nn.Linear(hidden_dim,num_items+1)

    def forward(self,item_seq,dow_seq,hour_seq):
        i_feat=self.item_emb(item_seq)
        d_feat=self.item_emb(dow_seq)
        h_feat=self.item_emb(hour_seq)

        combined=torch.cat([i_feat,d_feat,h_feat],dim=-1)
        x=self.merge(combined)

        _,(hidden,_)=self.lstm(x)
        x=self.fc(hidden[-1])
        return x