import math
import torch
import torch.nn as nn
import fsq

class PixelToStateAttention(nn.Module):
    def __init__(self, D, H, W, K, num_heads=1):
        super().__init__()
        # one query per state
        self.queries = nn.Parameter(torch.randn(K, D))
        self.mha     = nn.MultiheadAttention(embed_dim=D,
                                             num_heads=num_heads,
                                             batch_first=True)
    def forward(self, pixel_feats):
        # pixel_feats: (B, D, H, W)
        B,D,H,W = pixel_feats.shape
        S = H*W
        tokens = pixel_feats.view(B, D, S).permute(0,2,1)   # (B,S,D)
        Qs     = self.queries.unsqueeze(0).expand(B,-1,-1)  # (B,K,D)
        states, attn = self.mha(Qs, tokens, tokens,
                                need_weights=True,
                                average_attn_weights=False)
        # attn: (B, heads, K, S)
        return states, attn


class AECrossFSQ_SpaceTime_PixelState(AECrossFSQ):
    """
    Exactly the vanilla AECrossFSQ, plus a Tiny‐Conv + PixelToStateAttention
    *before* the linear encoder.  Encoding/decoding and FSQ are untouched.
    """
    def __init__(self, x_dim, y_dim, latent_size, alpha=1, lam=1, n_states=101,
                 num_heads=1):
        super().__init__(x_dim, y_dim, latent_size, alpha, lam, n_states)
        # assume x_dim, y_dim are perfect squares
        Hx = int(math.isqrt(x_dim)); Wx = Hx
        Hy = int(math.isqrt(y_dim)); Wy = Hy
        assert Hx*Wx==x_dim and Hy*Wy==y_dim

        self.Hx, self.Wx = Hx, Wx
        self.Hy, self.Wy = Hy, Wy

        # a tiny conv to lift raw input into D=latent_size tokens
        self.conv_x = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, latent_size, 3, padding=1), nn.ReLU()
        )
        self.conv_y = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, latent_size, 3, padding=1), nn.ReLU()
        )

        # attention queries per latent dimension
        self.p2s_x = PixelToStateAttention(latent_size, Hx, Wx,
                                           latent_size, num_heads)
        self.p2s_y = PixelToStateAttention(latent_size, Hy, Wy,
                                           latent_size, num_heads)

    def encode(self, x, y):
        B = x.size(0)
        # — INTERPRETABILITY PATH —
        # reshape raw x,y into images and compute attention maps
        with torch.no_grad():
            x_img = x.view(B,1,self.Hx,self.Wx)
            y_img = y.view(B,1,self.Hy,self.Wy)

            fx = self.conv_x(x_img)
            fy = self.conv_y(y_img)

            _, attn_x = self.p2s_x(fx)
            _, attn_y = self.p2s_y(fy)

            # store them for later
            self._last_attn_x = attn_x  # (B,heads,K,S)
            self._last_attn_y = attn_y

        # — VANILLA AE PATH (unchanged) —
        return super().encode(x, y)
    def forward(self, x, y):
        # run encode (will store attentions) and get quantized codes
        zx_q, zy_q = self.encode(x, y)
        # now run vanilla decoders
        xa = self.xx_decoder(zx_q)
        ya = self.yy_decoder(zy_q)
        xc = self.yx_decoder(zy_q)
        yc = self.xy_decoder(zx_q)
        return xa, ya, xc, yc, zx_q, zy_q

    def get_last_attentions(self):
        return getattr(self, '_last_attn_x', None), getattr(self, '_last_attn_y', None)
