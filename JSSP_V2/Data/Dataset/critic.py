# critic_copy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PtrNet2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding = nn.Linear(params["input_dim"], params["n_embedding"], bias=False)
        self.Encoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
        if torch.cuda.is_available():
            self.Vec = nn.Parameter(torch.cuda.FloatTensor(params["n_hidden"]))
        else:
            self.Vec = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
        self.W_q = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
        self.W_ref = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
        self.final2FC = nn.Sequential(
            nn.Linear(params["n_hidden"], params["n_hidden"], bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(params["n_hidden"], 1, bias=False))
        self._initialize_weights(params["init_min"], params["init_max"])
        self.n_glimpse = params["n_glimpse"]
        self.n_process = params["n_process"]

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, device):
        x = x.to(device)
        embed_enc_inputs = self.Embedding(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), -1)
        enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        ref = enc_h
        query = h[-1]

        for i in range(self.n_process):
            query = self.glimpse(query, ref)

        pred_l = self.final2FC(query).squeeze(-1)
        return pred_l

    def glimpse(self, query, ref, infinity=1e8):
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))
        u2 = self.W_ref(ref.permute(0, 2, 1))
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        a = F.softmax(u, dim=1)
        g = torch.bmm(a.unsqueeze(1), ref).squeeze(1)
        return g
