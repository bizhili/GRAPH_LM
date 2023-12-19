import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_length, output_dim, device= "cpu"):
        super(TransformerModel, self).__init__()

        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=3,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=128,
            device= device
        )
        self.input_dim= input_dim
        self.seq_length= seq_length

        self.fc = nn.Linear(seq_length * input_dim, output_dim, device= device)

    def forward(self, x):
        # Assuming x has shape (sequence_length, batch_size, input_dim)
        x = self.transformer(x, x)  # Use x as both source and target
        x = self.fc(x.view(x.shape[0], -1))
        return x