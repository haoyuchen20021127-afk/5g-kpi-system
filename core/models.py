"""The exact LSTM architecture used by the shipped checkpoint."""
from torch import nn


class JointLSTM(nn.Module):
    def __init__(self, input_size=4, output_size=4, hidden_size=32):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, history):
        if history.ndim != 3 or history.shape[2] != self.input_size:
            raise ValueError("输入必须为 [batch, time, input_size]。")
        sequence, _ = self.lstm(history)
        # Last observed hour summarizes the history; no future target is input.
        return self.head(sequence[:, -1, :])


def load_neural_checkpoint(checkpoint):
    if checkpoint.get('model_name', 'LSTM') != 'LSTM':
        raise ValueError('此工程仅支持 LSTM checkpoint。')
    config = checkpoint.get('config') or {'hidden_size': checkpoint['hidden_size']}
    model = JointLSTM(checkpoint['input_size'], checkpoint['output_size'], config['hidden_size'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model
