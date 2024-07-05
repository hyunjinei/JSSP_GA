# actor_copy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Greedy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.argmax(log_p, dim=1).long()

class Categorical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)

class PtrNet1(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding = nn.Linear(params["input_dim"], params["n_embedding"], bias=False)
        self.Encoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
        self.Decoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
        if torch.cuda.is_available():
            self.Vec = nn.Parameter(torch.cuda.FloatTensor(params["n_hidden"]))
            self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(params["n_hidden"]))
        else:
            self.Vec = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
            self.Vec2 = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
        self.W_q = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
        self.W_ref = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
        self.W_q2 = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
        self.W_ref2 = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
        self.dec_input = nn.Parameter(torch.FloatTensor(params["n_embedding"]))
        self._initialize_weights(params["init_min"], params["init_max"])
        self.use_logit_clipping = params["use_logit_clipping"]
        self.C = params["C"]
        self.T = params["T"]
        self.n_glimpse = params["n_glimpse"]
        self.block_selecter = {'greedy': Greedy(), 'sampling': Categorical()}.get(params["decode_type"], None)

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, device, y=None):
        print(f"Input shape before reshape: {x.size()}")
        x = x.to(device)

        # 만약 x가 3차원이라면 batch와 seq_len을 추출할 때 3차원을 고려
        if len(x.size()) == 3:
            batch, seq_len, input_dim = x.size()
            x = x.view(batch, seq_len, input_dim)
        else:
            batch, seq_len = x.size()

        # 임베딩 레이어에 맞는 입력 크기로 변환
        embed_enc_inputs = self.Embedding(x.view(-1, x.size(-1))).view(batch, seq_len, -1)
        print(f"Embedded input shape: {embed_enc_inputs.size()}")

        mask = torch.zeros((batch, seq_len), device=device)
        enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        ref = enc_h
        pi_list, log_ps = [], []
        dec_input = self.dec_input.unsqueeze(0).repeat(batch, 1).unsqueeze(1).to(device)

        for i in range(seq_len):
            _, (h, c) = self.Decoder(dec_input, (h, c))
            query = h.squeeze(0)
            for j in range(self.n_glimpse):
                query = self.glimpse(query, ref, mask)
            logits = self.pointer(query, ref, mask)
            log_p = torch.log_softmax(logits / self.T, dim=-1)

            if y is None:
                next_block = self.block_selecter(log_p)
            else:
                next_block = y[:, i].long()

            # next_block이 seq_len보다 작은지 확인하여 문제 방지
            next_block = torch.clamp(next_block, max=seq_len - 1)
            
            dec_input = torch.gather(input=embed_enc_inputs, dim=1,
                                    index=next_block.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed_enc_inputs.size(2)))
            print(f"dec_input shape: {dec_input.size()}")

            pi_list.append(next_block)
            log_ps.append(log_p)
            mask += torch.zeros((batch, seq_len), device=device).scatter_(dim=1, index=next_block.unsqueeze(1), value=1)

        pi = torch.stack(pi_list, dim=1)
        ps = torch.stack(log_ps, dim=1)
        ll = self.get_log_likelihood(torch.stack(log_ps, 1), pi)

        return pi, ll, ps


    def glimpse(self, query, ref, mask, inf=1e8):
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))
        u2 = self.W_ref(ref.permute(0, 2, 1))
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        u = u - inf * mask
        a = F.softmax(u, dim=1)
        g = torch.bmm(a.unsqueeze(1), ref).squeeze(1)
        return g

    def pointer(self, query, ref, mask, inf=1e8):
        u1 = self.W_q2(query).unsqueeze(-1).repeat(1, 1, ref.size(1))
        u2 = self.W_ref2(ref.permute(0, 2, 1))
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)

        if self.use_logit_clipping:
            u = self.C * torch.tanh(u)

        u = u - inf * mask
        return u

    def get_log_likelihood(self, _log_p, pi):
        log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
        return torch.sum(log_p.squeeze(-1), 1)

