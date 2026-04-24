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
        self.embedding_dim = embedding_dim

        self.user_embedding=nn.Embedding(num_users,embedding_dim)
        self.item_embedding=nn.Embedding(num_items+1,embedding_dim)

        self.dow_embedding=nn.Embedding(8,8)
        self.hour_embedding=nn.Embedding(25,8)

        nn.init.normal_(self.user_embedding.weight,std=0.01)
        nn.init.normal_(self.item_embedding.weight,std=0.01)
        
        self.mlp=nn.Sequential(
            nn.Linear(embedding_dim*4, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,user,item,dow,hour):
        u_emb=self.user_embedding(user)
        i_emb=self.item_embedding(item)
        d_emb=self.item_embedding(dow)
        h_emb=self.item_embedding(hour)
        x=torch.cat([u_emb,i_emb,d_emb,h_emb],dim=1)
        x=self.mlp(x)
        return x.squeeze()