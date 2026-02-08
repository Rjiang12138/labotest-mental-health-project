import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.device = args.device
        self.embedding = nn.Embedding(args.n_vocab, args.embed)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, args.num_filters, (k, args.embed)) for k in args.filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.num_filters * len(args.filter_sizes), args.num_class)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = F.normalize(x.float(), p=1, dim=1).long()
        out = self.embedding(x).to(self.device)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
