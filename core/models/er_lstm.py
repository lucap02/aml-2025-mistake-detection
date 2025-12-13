import torch
from torch import nn

from core.models.blocks import fetch_input_dim, MLP


class ErLSTM(nn.Module):
    """
    LSTM-based baseline for SupervisedER (Step-level)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        input_dim = fetch_input_dim(config)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_dim = 256 * 2  # because bidirectional

        self.decoder = MLP(lstm_out_dim, 512, 1)

    def forward(self, input_data):
        """
        input_data: (B, T, D) where B is batch size, T is sequence length, D is feature dimension
        """

        input_data = torch.nan_to_num(
            input_data,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        lstm_out, (h_n, _) = self.lstm(input_data)

        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)

        final_output = self.decoder(h_last)

        return final_output

