import torch
import torch.optim as optim
import torch.nn as nn
from GraphData import GraphDataset
from GraphData_solution import JSSPDataset, load_solutions
from Transformer_Graph import TransformerModel, train_model, evaluate_model
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    graph_dataset = GraphDataset('la02.txt')
    graphs = [graph_dataset.graph for _ in range(100)]  # 예시로 100개의 그래프 데이터 복제
    solutions = load_solutions('la02_solutions.txt')  # 솔루션 데이터 로드

    # 데이터셋 생성
    dataset = JSSPDataset(graphs, solutions, graph_dataset.n_machine)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # 검증 데이터셋 분할
    val_split = 0.1
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    # 모델 생성
    input_dim = 6
    model_dim = 128
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 512
    output_dim = 50  # Operation 수
    dropout = 0.1
    model = TransformerModel(input_dim, model_dim, num_encoder_layers, num_decoder_layers, dim_feedforward, output_dim, dropout).to(device)

    # 옵티마이저와 손실 함수 정의
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 학습 및 검증
    epochs = 10
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_model(model, train_dataloader, optimizer, criterion, device)
        train_loss = evaluate_model(model, train_dataloader, device)  # device 전달
        val_loss = evaluate_model(model, val_dataloader, device)  # device 전달
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # 손실 값 시각화
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 모델 저장
    torch.save(model.state_dict(), 'transformer_model.pth')

if __name__ == "__main__":
    main()
