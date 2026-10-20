import torch
import torch.nn as nn

class SignLSTM(nn.Module):
    """
    Bidirectional LSTM with Dropout and LayerNorm for multi-class ISL sentence recognition.
    """
    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=50, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Bidirectional outputs 2 * hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the last time step output or mean pooling across time
        # Mean pooling across sequence for stable sentence representation
        feat = torch.mean(lstm_out, dim=1)
        feat = self.layer_norm(feat)
        feat = self.dropout(feat)
        
        feat = self.relu(self.fc1(feat))
        feat = self.dropout(feat)
        logits = self.fc2(feat)
        return logits
