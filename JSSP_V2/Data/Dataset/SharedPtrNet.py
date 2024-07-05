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

class SharedPtrNet(nn.Module):
    def __init__(self, shared_params, dynamic_params):
        super().__init__()
        self.shared_embedding = nn.Linear(dynamic_params["embedding_input_dim"], shared_params["n_embedding"], bias=False)
        self.Encoder = nn.LSTM(input_size=shared_params["n_embedding"], hidden_size=shared_params["n_hidden"], batch_first=True)
        self.Decoder = nn.LSTM(input_size=shared_params["n_embedding"], hidden_size=shared_params["n_hidden"], batch_first=True)
        if torch.cuda.is_available():
            self.Vec = nn.Parameter(torch.cuda.FloatTensor(shared_params["n_hidden"]))
            self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(shared_params["n_hidden"]))
        else:
            self.Vec = nn.Parameter(torch.FloatTensor(shared_params["n_hidden"]))
            self.Vec2 = nn.Parameter(torch.FloatTensor(shared_params["n_hidden"]))
        self.W_q = nn.Linear(shared_params["n_hidden"], shared_params["n_hidden"], bias=True)
        self.W_ref = nn.Conv1d(shared_params["n_hidden"], shared_params["n_hidden"], 1, 1)
        self.W_q2 = nn.Linear(shared_params["n_hidden"], shared_params["n_hidden"], bias=True)
        self.W_ref2 = nn.Conv1d(shared_params["n_hidden"], shared_params["n_hidden"], 1, 1)
        self.dec_input = nn.Parameter(torch.FloatTensor(shared_params["n_embedding"]))
        self._initialize_weights(shared_params["init_min"], shared_params["init_max"])
        self.use_logit_clipping = shared_params["use_logit_clipping"]
        self.C = shared_params["C"]
        self.T = shared_params["T"]
        self.n_glimpse = shared_params["n_glimpse"]
        self.block_selecter = {'greedy': Greedy(), 'sampling': Categorical()}.get(shared_params["decode_type"], None)

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, device, action=None):
        logging.info(f"Input shape before reshape: {x.size()}")
        logging.info(f"Input dtype before reshape: {x.dtype}")
        x = x.float()
        logging.info(f"Input shape after converting to float: {x.size()}")
        
        embed_inputs = self.encoder_embedding(x)
        logging.info(f"Shared embedding input shape: {embed_inputs.size()}")
        
        # Shared layers
        embed_enc_inputs = self.shared_encoder_embedding(embed_inputs)
        logging.info(f"Embedded input shape: {embed_enc_inputs.size()}")
        
        dec_input = embed_enc_inputs[:, -1, :].unsqueeze(1)
        logging.info(f"dec_input shape: {dec_input.size()}")
        
        batch, input_dim = x.size()
        x = x.view(batch, 1, input_dim)
        
        embed_enc_inputs = self.shared_encoder_embedding(x)
        logging.info(f"Embedded input shape: {embed_enc_inputs.size()}")
        
        dec_input = embed_enc_inputs[:, -1, :].unsqueeze(1)
        logging.info(f"dec_input shape: {dec_input.size()}")
        
        ref = embed_enc_inputs
        query = dec_input
        mask = torch.zeros(ref.size()[:2], dtype=torch.bool, device=device)
        
        for i in range(self.n_process):
            query = self.glimpse(query, ref, mask)
        
        pred_l = self.final2FC(query).squeeze(-1)
        return pred_l, None, None

    # SharedPtrNet 클래스의 일부
    def glimpse(self, query, ref, mask, infinity=1e8):
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))
        u2 = self.W_ref(ref.permute(0, 2, 1))
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        u = u - infinity * (~mask).float()  # 부울 마스크 연산 수정
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
