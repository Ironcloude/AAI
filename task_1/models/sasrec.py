import torch
import torch.nn as nn
import math

class SASRec(nn.Module):
    def __init__(self, num_items, embedding_dim=32, num_heads=2, num_layers=2, max_len=50, dropout_rate=0.1):
        super(SASRec, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_len = max_len

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)

        self.dow_embedding=nn.Embedding(8,embedding_dim,padding_idx=0)
        self.hour_embedding=nn.Embedding(25,embedding_dim,padding_idx=0)

        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.feature_fusion=nn.Linear(embedding_dim*3,embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim*4,
            batch_first=True,
            dropout=dropout_rate,
            norm_first=True
        )
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_items + 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, item_seq, dow_seq, hour_seq):
        device=item_seq.device
        batch_size, seq_len = item_seq.size()
        
        i_emb=self.item_embedding(item_seq)
        d_emb=self.item_embedding(dow_seq)
        h_emb=self.item_embedding(hour_seq)

        i_emb=i_emb*math.sqrt(self.embedding_dim)

        combined_features=torch.cat([i_emb,d_emb,h_emb],dim=-1)
        x=self.feature_fusion(combined_features)
    
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = x + self.position_embedding(positions)
        x = self.emb_dropout(x)

        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()

        att_output=self.transformer_encoder(x,mask=causal_mask)
        last_output=att_output[:,-1,:]
        logits=self.fc(last_output)

        #x = self.layer_norm(x)

        
        return logits