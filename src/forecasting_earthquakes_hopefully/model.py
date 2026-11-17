"""The model of the normal days: one station's day predicted from the other stations of its region.

Input   the band envelopes at 1 Hz of the other stations on the same day (envelopes.py), each band on the
        scale of its station, plus a 0/1 mask of the seconds with data
Output  for the station that is left out, per minute: the mean and the maximum of each band on that station's
        scale. Its own recording of the day, and of the days before, is not given to the model: what it knows
        of the station is a learned vector for it, the minute of the day and the day of the week.

So the prediction holds what the station usually does at that time and what its neighbours see that day
(weather, distant earthquakes). What is left over (observed minus predicted) is what only this station saw.

  tokenizer   two convolutions at 1 s, then the mean and the maximum of every 60 s side by side (a spike of a
              few seconds survives in the maximum), then a convolution over 5 minutes: 1440 tokens a day
  in a day    a bidirectional GRU over the tokens of each station
  across      per minute, attention from the query of the left-out station over the tokens of the others
  out         a bidirectional GRU over the minutes and a linear layer to 2 x bands numbers
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

MINUTES = 1440
SECONDS = 86400


class Tokenizer(nn.Module):
    def __init__(self, bands, width):
        super().__init__()
        self.c1 = nn.Conv1d(bands + 1, 32, 9, padding=4)
        self.c2 = nn.Conv1d(32, 32, 9, padding=4)
        self.mix = nn.Conv1d(64, width, 5, padding=2)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):                                   # (N, bands + 1, 86400)
        h = F.gelu(self.c2(F.gelu(self.c1(x))))
        h = torch.cat([F.avg_pool1d(h, 60), F.max_pool1d(h, 60)], dim=1)
        return F.gelu(self.norm(self.mix(h)))               # (N, width, 1440)


class Normal(nn.Module):
    def __init__(self, stations, bands=5, width=64, heads=4):
        super().__init__()
        self.tokenizer = Tokenizer(bands, width)
        self.day = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        self.station = nn.Embedding(stations, width)
        self.when = nn.Linear(4 + 7, width)
        self.across = nn.MultiheadAttention(width, heads, batch_first=True)
        self.minutes = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(width, 2 * bands)
        minute = torch.arange(MINUTES) / MINUTES * 2 * math.pi
        self.register_buffer("clock", torch.stack([minute.sin(), minute.cos(), (2 * minute).sin(), (2 * minute).cos()], dim=1))

    def forward(self, x, present, target, weekday):
        """x (B, S, bands + 1, 86400) with the stations in a fixed order; present (B, S) bool, False for a
        station without the day and for the target; target (B,) index of the station left out; weekday (B,).
        Returns (B, 1440, 2 * bands): per band the mean, then per band the maximum."""
        B, S = x.shape[:2]
        tok = self.tokenizer(x.flatten(0, 1)).transpose(1, 2)                       # (B * S, 1440, W)
        tok = self.day(tok)[0].reshape(B, S, MINUTES, -1) + self.station.weight[None, :S, None]
        when = torch.cat([self.clock[None].expand(B, -1, -1),
                          F.one_hot(weekday, 7).float()[:, None].expand(-1, MINUTES, -1)], dim=2)
        query = self.station(target)[:, None] + self.when(when)                     # (B, 1440, W)
        keys = tok.permute(0, 2, 1, 3).reshape(B * MINUTES, S, -1)
        hidden = (~present)[:, None].expand(-1, MINUTES, -1).reshape(B * MINUTES, S)
        seen = self.across(query.reshape(B * MINUTES, 1, -1), keys, keys, key_padding_mask=hidden)[0]
        h = seen.reshape(B, MINUTES, -1) + query
        return self.head(self.minutes(h)[0])
