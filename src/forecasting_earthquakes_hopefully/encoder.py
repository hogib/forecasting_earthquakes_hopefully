"""The models of the classifier.

  DayEncoder     one day of per-minute summaries (windows.py) -> SLOTS tokens of SLOT minutes -> one vector.
                 Three convolutions bring the 1440 minutes down to 96 quarter hours, a bidirectional GRU reads
                 them; the vector is the mean and the maximum of the tokens through a linear layer.
  Reconstructor  the pretraining without labels: a share of the quarter hours is hidden from the encoder and
                 a linear layer on each token has to give back the hidden minutes.
  Classifier     a GRU over the vectors of the days of a window and one number out: the logit of "an earthquake
                 of the question follows within the lead time".
"""
import torch
from torch import nn
from torch.nn import functional as F

MINUTES = 1440
SLOT = 15
SLOTS = MINUTES // SLOT


class DayEncoder(nn.Module):
    def __init__(self, features, components, width=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(features + components + 1, width, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(width, width, 3, stride=3), nn.GELU(),
                                  nn.Conv1d(width, width, 5, stride=5), nn.GELU(), nn.GroupNorm(8, width))
        self.gru = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * width, width)

    def tokens(self, z, mask, hidden=None):
        """z (N, features, 1440), mask (N, components, 1440), hidden (N, SLOTS) bool or None -> (N, SLOTS, width)."""
        if hidden is None:
            h = z.new_zeros(z.shape[0], 1, MINUTES)
        else:
            h = hidden.repeat_interleave(SLOT, dim=1)[:, None].to(z.dtype)
            z, mask = z * (1 - h), mask * (1 - h)
        return self.gru(self.stem(torch.cat([z, mask, h], dim=1)).transpose(1, 2))[0]

    def forward(self, z, mask):
        t = self.tokens(z, mask)
        return self.out(torch.cat([t.mean(dim=1), t.amax(dim=1)], dim=1))


class Reconstructor(nn.Module):
    def __init__(self, encoder, features, comp_of, width=64):
        super().__init__()
        self.encoder, self.features = encoder, features
        self.head = nn.Linear(width, SLOT * features)
        self.register_buffer("comp_of", torch.as_tensor(comp_of))

    def loss(self, z, mask, share=0.25, generator=None):
        hidden = torch.rand(z.shape[0], SLOTS, device=z.device, generator=generator) < share
        pred = self.head(self.encoder.tokens(z, mask, hidden))                               # (N, SLOTS, SLOT * F)
        pred = pred.reshape(-1, SLOTS, self.features, SLOT).permute(0, 2, 1, 3).reshape(-1, self.features, MINUTES)
        judged = (hidden.repeat_interleave(SLOT, dim=1)[:, None] & (mask[:, self.comp_of] > 0))
        if not judged.any():
            return pred.sum() * 0
        return F.huber_loss(pred[judged], z[judged])


class Classifier(nn.Module):
    def __init__(self, width=64, hidden=32, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(width + 1, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, v, present):
        """v (B, days, width) the day vectors, present (B, days) 0/1 -> (B,) logits."""
        p = present[..., None].to(v.dtype)
        return self.head(self.drop(self.gru(torch.cat([v * p, p], dim=-1))[0][:, -1])).squeeze(-1)
