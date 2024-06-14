import torch
import torch.nn as nn
from GraphData import GraphDataset
from GraphData_solution import JSSPDataset
from Transformer_Graph import TransformerModel
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_model(model_path, input_dim, model_dim, num_encoder_layers, num_decoder_layers, dim_feedforward, output_dim, dropout, device):
    model = TransformerModel(input_dim, model_dim, num_encoder_layers, num_decoder_layers, dim_feedforward, output_dim, dropout).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, dataset, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch_idx, (node_tensor, edge_index_tensor, _) in enumerate(dataloader):
            src = node_tensor.to(device)
            tgt = torch.zeros((src.size(0), src.size(1), src.size(2))).to(device)  # src와 같은 크기로 초기화
            print(f"Forward called with src shape: {src.shape}, tgt shape: {tgt.shape}")
            src_emb = model.embedding(src)
            tgt_emb = model.embedding(tgt)
            src_emb = src_emb.permute(1, 0, 2)  # [seq_len, batch_size, model_dim]
            tgt_emb = tgt_emb.permute(1, 0, 2)  # [seq_len, batch_size, model_dim]
            output = model.transformer(src_emb, tgt_emb)
            output = model.fc_out(output)
            print(f"output shape: {output.shape}")
            predictions.append(output.argmax(dim=-1).cpu().numpy().tolist())
    return predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    graph_dataset = GraphDataset('la02.txt')
    graphs = [graph_dataset.graph]  # 예측을 위한 단일 그래프 데이터
    dummy_solutions = [[0] * graph_dataset.n_machine * graph_dataset.n_job]  # 임시 솔루션 데이터

    # 데이터셋 생성
    dataset = JSSPDataset(graphs, dummy_solutions, graph_dataset.n_machine)

    # 모델 로드
    model_path = 'transformer_model.pth'
    input_dim = 6
    model_dim = 128
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 512
    output_dim = 50  # Operation 수
    dropout = 0.1
    model = load_model(model_path, input_dim, model_dim, num_encoder_layers, num_decoder_layers, dim_feedforward, output_dim, dropout, device)

    # 예측 수행
    predictions = predict(model, dataset, device)
    for i, prediction in enumerate(predictions):
        print(f"Prediction for sample {i}: {prediction}")

if __name__ == "__main__":
    main()
