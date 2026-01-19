import torch.nn as nn

class TemporalLSTM(nn.Module):
    """
    Temporal LSTM Module
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,   # Must match the input feature size
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.hidden_dim = hidden_dim

    def forward(self, x_seq):
        # x_seq: [B, Seq, input_dim]
        lstm_out, _ = self.lstm(x_seq)  # [B, Seq, hidden_dim]
        return lstm_out[:, -1, :]       # Take last timestep output
