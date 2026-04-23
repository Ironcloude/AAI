import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF)
    Input: user_id, item_id
    Output: interaction score
    """
    def __init__(self,num_users,num_items,embedding_dim=32):
        super().__init__()
        self.user_embedding=nn.Embedding(num_users,embedding_dim)
        self.item_embedding=nn.Embedding(num_items,embedding_dim)

        self.mlp=nn.Sequential(
            nn.Linear(embedding_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,user,item):
        user_emb=self.user_embedding(user)
        item_emb=self.item_embedding(item)
        x=torch.cat([user_emb,item_emb],dim=1)
        x=self.mlp(x)
        return x.squeeze()