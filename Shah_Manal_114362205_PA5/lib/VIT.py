import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CountingPositionEmbs(nn.Module):
    def __init__(self, Max_num_patches, emb_dim, dropout_rate=0.1):
        super(CountingPositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, Max_num_patches, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        patches = x.shape[0]
        pos_embedding = self.pos_embedding.squeeze()
        out = x + pos_embedding[0:patches]

        if self.dropout:
            out = self.dropout(out)
        out = out.unsqueeze(0)
        return out


class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        """
            PART I: 20 points
            FC layer(dropout) + GELU act layer + FC layer(dropout)
        """
        """ STUDENT CODE START """
        #FC Layer
        self.fc1 = nn.Linear(in_dim, mlp_dim, bias=True)
        #Dropout 1
        self.dp1 = nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0.0 else None
        #GELU Activation
        self.activation_layer = nn.GELU()
        #FC Layer
        self.fc2 = nn.Linear(mlp_dim, out_dim, bias=True)
        #Dropout 2
        self.dp2 = nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0.0 else None
        """ STUDENT CODE END """

    def forward(self, x):

        """ STUDENT CODE START """
        out = self.fc1(x)
        out = self.activation_layer(out)
        if self.dp1:
          out = self.dp1(out)
        out = self.fc2(out)
        if self.dp2:
          out = self.dp2(out)
        """ STUDENT CODE END """
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, scale):
        scores = query.matmul(key.transpose(-2, -1)) / scale
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)

# Multi-head attention
class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()

        """
            PART II: 40 points
            multihead SelfAttention part with Key, Query, Value
            use LinearGeneral class
        """
        """ STUDENT CODE BEGIN """
        feat_h, feat_w = heads, int(in_dim / heads)

        self.scale = pow(feat_w, 0.5)
        
        #Query, Key, Value
        self.linear_query  = LinearGeneral((in_dim,), (feat_h, feat_w))
        self.linear_key    = LinearGeneral((in_dim,), (feat_h, feat_w))
        self.linear_value  = LinearGeneral((in_dim,), (feat_h, feat_w)) 
        #Out
        self.linear_out    = LinearGeneral((feat_h, feat_w), (in_dim,))
        """ STUDENT CODE END"""

    def forward(self, x):

        """ STUDENT CODE BEGIN """
        dims, permutation_order = ([2], [0]), (0, 2, 1, 3)
        q, k, v = self.linear_query(x, dims=dims), self.linear_key(x, dims=dims), self.linear_value(x, dims=dims)
        q, k, v = q.permute(permutation_order), k.permute(permutation_order), v.permute(permutation_order)

        out = ScaledDotProductAttention()(q, k, v, self.scale)
        out = out.permute(permutation_order)

        out = self.linear_out(out, dims=([2, 3], [0, 1]))

        """ STUDENT CODE END """
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out

# 12 Transformer layers
class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        """
            PART III: 20 points
            Apply 12 layer of transformer and one LyaerNorm
        """
        """ STUDENT CODE START """
        # 12 Transformer Layers
        for i in range(num_layers):
            self.encoder_layers.append(EncoderBlock(in_dim, 
                                                    mlp_dim, 
                                                    num_heads, 
                                                    dropout_rate, 
                                                    attn_dropout_rate))
        # 1 Layer Norm
        self.norm = nn.LayerNorm(in_dim)
        """ STUDENT CODE END """


    def forward(self, x):

        out = self.pos_embedding(x)

        """ STUDENT CODE START """
        for layer in self.encoder_layers:
            out = layer(out)
        out = self.norm(out)
        """ STUDENT CODE END """
        return out

class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self,
                 image_size=(256, 256),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 feat_dim=None):
        super(VisionTransformer, self).__init__()
        h, w = image_size

        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        # classfier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        print(x.shape)
        emb = self.embedding(x)  # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        # transformer
        feat = self.transformer(emb)

        # classifier
        logits = self.classifier(feat[:, 0])
        return logits
