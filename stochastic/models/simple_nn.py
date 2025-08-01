import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size=28*28, output_size=10, hidden_layers=[128, 64, 32, 32]):
        super(SimpleNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        layers = []
        prev_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))
    
    def __repr__(self):
        return f"Simple NN (ReLU activations) with {self.input_size=}, {self.output_size=}, {self.hidden_layers=}"