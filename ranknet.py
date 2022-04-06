from unicodedata import bidirectional
import torch
from torch import nn


class RankNet(nn.Module):
    def __init__(
        self,
        embedding_size: int = 256,
        hidden_size: int = 128,
        bidirectional: bool = True,
        hidden_layers: int = 1,
        dropout_rate: float = 0.5,
        cell_type: str = 'GRU'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        if cell_type.upper() == 'GRU':
            cell_class = nn.GRU
        elif cell_type.upper() == 'LSTM':
            cell_class = nn.LSTM
        else:
            raise NotImplementedError(
                f"'{cell_type}' is not implemented. please use one of 'GRU' and 'LSTM' for cell type")
        self.lstm = cell_class(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional,
        )

        self.drop = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

        linear_size = hidden_size
        if bidirectional:
            linear_size = linear_size * 2
        self.dense = nn.Sequential(
            nn.Linear(linear_size, 1),
            
        )

    def forward_one(self, inp):
        out, _ = self.lstm(inp)
        # out = out[:,-1,:]
        if self.bidirectional:
            forward = out[:, -1, : self.hidden_size]
            backward = out[:, 0, self.hidden_size:]
            out = torch.cat((forward, backward), dim=1)
        else:
            out = out[:,-1,:]
        out = self.relu(out)
        out = self.drop(out)
        out = self.dense(out)

        return out

    def forward(self, inp1: torch.Tensor, inp2: torch.Tensor):
        out1 = self.forward_one(inp1) # coronavirus origin asian -> 0.01
        out2 = self.forward_one(inp2) # coronavirus origin spread -> 0.5
        return (
            out1 - out2 # 0.01 - 0.5 = -0.49 
        )

    def predict(self, inp: torch.Tensor):
        return self.forward_one(inp) # coronavirus origin asian -> 0.9
        # coronavirus origin spread -> 0.001


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    r = RankNet(bidirectional=False, hidden_size=64, hidden_layers=1)
    print(count_parameters(r))
