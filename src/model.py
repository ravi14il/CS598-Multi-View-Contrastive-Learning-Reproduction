import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * 
                             (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, hidden_dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class InteractionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(InteractionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, ht, hd, hf):
        N, L, D = ht.size()
        h = torch.stack([ht, hd, hf], dim=2)  # [N, L, 3, D]
        h = h.permute(0, 2, 1, 3).contiguous().view(N * 3, L, D)  # [N*3, L, D]
    
        # Apply attention across the time dimension
        attn_output, _ = self.multihead_attn(h, h, h)
        output = self.norm(h + attn_output)
        output = output.view(N, 3, L, D).permute(0, 2, 1, 3)  # [N, L, 3, D]
    
        ht_i, hd_i, hf_i = output[:, :, 0, :], output[:, :, 1, :], output[:, :, 2, :]
        return ht_i, hd_i, hf_i

        
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.extension = getattr(args, "extension", None)
        # Use CLS tokens only if extension includes "CLS"
        self.use_cls = self.extension in ["CLS", "CLS-Weight"]

        ## initial encoding
        self.positional_encoding = PositionalEncoding(
            args.num_embedding, args.dropout
        )

        # per-view CLS tokens
        if self.use_cls:
            self.cls_t = nn.Parameter(torch.zeros(1, 1, args.num_embedding))
            self.cls_d = nn.Parameter(torch.zeros(1, 1, args.num_embedding))
            self.cls_f = nn.Parameter(torch.zeros(1, 1, args.num_embedding))
            nn.init.trunc_normal_(self.cls_t, std=0.02)
            nn.init.trunc_normal_(self.cls_d, std=0.02)
            nn.init.trunc_normal_(self.cls_f, std=0.02)

        # view-specific encoders
        self.input_layer_t = nn.Linear(args.num_feature, args.num_embedding)
        self.encoder_layers_t = nn.TransformerEncoderLayer(d_model=args.num_embedding, dim_feedforward=args.num_hidden, nhead=args.num_head, dropout=args.dropout, batch_first=True)
        self.transformer_encoder_t = nn.TransformerEncoder(self.encoder_layers_t, args.num_layers)        

        self.input_layer_d = nn.Linear(args.num_feature, args.num_embedding)
        self.encoder_layers_d = nn.TransformerEncoderLayer(d_model=args.num_embedding, dim_feedforward=args.num_hidden, nhead=args.num_head, dropout=args.dropout, batch_first=True)
        self.transformer_encoder_d = nn.TransformerEncoder(self.encoder_layers_d, args.num_layers)

        self.input_layer_f = nn.Linear(args.num_feature, args.num_embedding)
        self.encoder_layers_f = nn.TransformerEncoderLayer(
            d_model=args.num_embedding,
            dim_feedforward=args.num_hidden,
            nhead=args.num_head,
            dropout=args.dropout,
            batch_first=True,
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            self.encoder_layers_f, args.num_layers
        )

        ## interaction
        self.interaction_layer = InteractionLayer(args.num_embedding, args.num_head)
        
        ## output
        self.output_layer_t = nn.Sequential(
            nn.Linear(args.num_embedding*2, args.num_hidden),
            nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden)
        )
        self.output_layer_d = nn.Sequential(
            nn.Linear(args.num_embedding*2, args.num_hidden),
            nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden)
        )
        self.output_layer_f = nn.Sequential(
            nn.Linear(args.num_embedding*2, args.num_hidden),
            nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden)
        )
    
    def forward(self, xt, dx, xf):
        xt, dx, xf = torch.nan_to_num(xt), torch.nan_to_num(dx), torch.nan_to_num(xf)

        if self.use_cls:
            B = xt.size(0)

            # ----- time view -----
            ht = self.input_layer_t(xt)             # [B, L, D]
            cls_t = self.cls_t.expand(B, 1, -1)     # [B, 1, D]
            ht = torch.cat([cls_t, ht], dim=1)      # [B, L+1, D]
            ht = self.positional_encoding(ht)
            ht = self.transformer_encoder_t(ht)     # [B, L+1, D]
            ht_cls = ht[:, 0, :]                    # [B, D]
            ht_tok = ht[:, 1:, :]                   # [B, L, D]

            # ----- derivative view -----
            hd = self.input_layer_d(dx)
            cls_d = self.cls_d.expand(B, 1, -1)
            hd = torch.cat([cls_d, hd], dim=1)
            hd = self.positional_encoding(hd)
            hd = self.transformer_encoder_d(hd)
            hd_cls = hd[:, 0, :]
            hd_tok = hd[:, 1:, :]

            # ----- frequency view -----
            hf = self.input_layer_f(xf)
            cls_f = self.cls_f.expand(B, 1, -1)
            hf = torch.cat([cls_f, hf], dim=1)
            hf = self.positional_encoding(hf)
            hf = self.transformer_encoder_f(hf)
            hf_cls = hf[:, 0, :]
            hf_tok = hf[:, 1:, :]

            # interaction on tokens only (no CLS)
            ht_i_tok, hd_i_tok, hf_i_tok = self.interaction_layer(
                ht_tok, hd_tok, hf_tok
            )  # [B, L, D] each

            # pool the fused tokens
            ht_i_pool = ht_i_tok.mean(dim=1)   # [B, D]
            hd_i_pool = hd_i_tok.mean(dim=1)   # [B, D]
            hf_i_pool = hf_i_tok.mean(dim=1)   # [B, D]

            # final per-view embeddings: [CLS || pooled interaction]
            zt = self.output_layer_t(torch.cat([ht_cls, ht_i_pool], dim=-1))
            zd = self.output_layer_d(torch.cat([hd_cls, hd_i_pool], dim=-1))
            zf = self.output_layer_f(torch.cat([hf_cls, hf_i_pool], dim=-1))

            # ht, hd, hf still include CLS at index 0
            return ht, hd, hf, zt, zd, zf


        ht = self.input_layer_t(xt)
        ht = self.positional_encoding(ht)
        ht = self.transformer_encoder_t(ht)
        
        hd = self.input_layer_d(dx)
        hd = self.positional_encoding(hd)
        hd = self.transformer_encoder_d(hd)

        hf = self.input_layer_f(xf)
        hf = self.positional_encoding(hf)
        hf = self.transformer_encoder_f(hf)

        # interaction
        ht_i, hd_i, hf_i = self.interaction_layer(ht, hd, hf)
        
        # output layers (mean pooling as in the original paper)
        zt = self.output_layer_t(
            torch.cat([ht.mean(dim=1), ht_i.mean(dim=1)], dim=-1)
        )
        zd = self.output_layer_d(
            torch.cat([hd.mean(dim=1), hd_i.mean(dim=1)], dim=-1)
        )
        zf = self.output_layer_f(
            torch.cat([hf.mean(dim=1), hf_i.mean(dim=1)], dim=-1)
        )

        return ht, hd, hf, zt, zd, zf


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        q = self.query(x)  # (batch_size, 3, hidden_dim)
        k = self.key(x)    # (batch_size, 3, hidden_dim)
        v = self.value(x)  # (batch_size, 3, hidden_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # (batch_size, 3, 3)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, 3, 3)
        output = torch.matmul(attention_weights, v)  # (batch_size, 3, hidden_dim)
        return output, attention_weights
        

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.extension = getattr(args, "extension", None)
        # static view weights only if extension includes "Weight"
        self.use_view_weights = self.extension in ["Weight", "CLS-Weight"]

        if self.args.feature == 'latent':
            if args.loss_type == 'ALL':
                self.self_attention = SelfAttention(args.num_hidden)
                if self.use_view_weights:
                    # logit space parameters for (T,D,F)
                    self.view_logit = nn.Parameter(torch.zeros(3))

        elif self.args.feature == 'hidden':
            ## interaction
            self.interaction_layer = InteractionLayer(args.num_embedding, args.num_head)
            
            ## output
            self.output_layer_t = nn.Sequential(
                nn.Linear(args.num_embedding*2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden)
            )
            self.output_layer_d = nn.Sequential(
                nn.Linear(args.num_embedding*2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden)
            )
            self.output_layer_f = nn.Sequential(
                nn.Linear(args.num_embedding*2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden)
            )

        self.fc = nn.Linear(len(args.loss_type)*args.num_hidden, args.num_target)
        
        self.fc.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, xt, dx, xf):
        xt, dx, xf = torch.nan_to_num(xt), torch.nan_to_num(dx), torch.nan_to_num(xf)

        if self.args.feature == 'latent':
            zt, zd, zf = xt, dx, xf

            if self.args.loss_type == 'ALL':
                # base self-attention fusion
                stacked = torch.stack([zt, zd, zf], dim=1)  # [B, 3, H]
                attn_out, _ = self.self_attention(stacked)
                emb_views = attn_out + stacked              # [B, 3, H]
                zt, zd, zf = (
                    emb_views[:, 0, :],
                    emb_views[:, 1, :],
                    emb_views[:, 2, :],
                )

                # optional static view weights (T,D,F)
                if self.use_view_weights:
                    w = torch.softmax(self.view_logit, dim=0)   # [3]
                    zt = zt * w[0]
                    zd = zd * w[1]
                    zf = zf * w[2]
        
        elif self.args.feature == 'hidden':
            ht, hd, hf = xt, dx, xf
    
            # interaction
            if self.args.loss_type == 'ALL':
                ht_i, hd_i, hf_i = self.interaction_layer(ht, hd, hf)
            else:
                ht_i, hd_i, hf_i = ht, hd, hf
            
            # output layers
            zt = self.output_layer_t(torch.cat([ht.mean(dim=1), ht_i.mean(dim=1)], dim=-1))
            zd = self.output_layer_d(torch.cat([hd.mean(dim=1), hd_i.mean(dim=1)], dim=-1))
            zf = self.output_layer_f(torch.cat([hf.mean(dim=1), hf_i.mean(dim=1)], dim=-1))
        
        if self.args.loss_type == 'ALL':
            emb = torch.cat([zt, zd, zf], dim=-1)
        else:
            emb_list = []
            # append embeddings based on the selected loss type
            if ('T' in self.args.loss_type):
                emb_list.append(zt)
            if ('D' in self.args.loss_type):
                emb_list.append(zd)
            if ('F' in self.args.loss_type):
                emb_list.append(zf)
            emb = torch.cat(emb_list, dim=-1)
            
        emb = emb.reshape(emb.shape[0], -1)
        return self.fc(emb)
