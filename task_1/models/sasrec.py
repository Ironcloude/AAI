import torch
import torch.nn as nn

class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommender (NCF)
    """
    def __init__(self, num_items, embedding_dim=32, num_heads=2, num_layers=2, max_len=50):
        super().__init__()

        self.item_embedding=nn.Embedding(num_items,embedding_dim)
        self.position_embedding=nn.Embedding(max_len,embedding_dim)

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer=nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.fc=nn.Linear(embedding_dim,num_items)
        self.max_len=max_len

    def forward(self, seq):
        batch_size,seq_len=seq.shape
        positions=torch.arange(seq_len,device=seq.device).unsqueeze(0).expand(batch_size,seq_len)
        x=self.item_embedding(seq)+self.position_embedding(positions)
        x=self.transformer(x)
        x=x[:,-1,:]
        x=self.fc(x)
        return x