import torch
from torch import nn

from core.models.blocks import fetch_input_dim, MLP


class ErLSTM(nn.Module):
    """
    LSTM-based baseline per Step-level Error Recognition.
    Restituisce un output per ogni step della sequenza.
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

        lstm_out_dim = 256 * 2  # bidirezionale

        # Decoder applicato ad ogni step
        self.decoder = MLP(lstm_out_dim, 512, 1)

    def forward(self, input_data):
        """
        input_data: (B, T, D)
            B = batch size
            T = lunghezza sequenza
            D = dimensione features
        output: (B, T, 1)
        """
        # Gestione NaN e infiniti
        input_data = torch.nan_to_num(
            input_data,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        # LSTM: lstm_out -> (B, T, H*2)
        lstm_out, _ = self.lstm(input_data)

        # Decoder applicato ad ogni step
        final_output = self.decoder(lstm_out)  # (B, T, 1)

        return final_output
