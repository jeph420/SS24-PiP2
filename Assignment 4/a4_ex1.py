import torch

class SimpleNetwork(torch.nn.Module):

    def __init__(self,
                 input_neurons: int, hidden_neurons: list, output_neurons: int,
                 use_bias: bool, activation_function: torch.nn.Module = torch.nn.ReLU()):
        
        super().__init__()
        self.input = input_neurons
        self.hidden = hidden_neurons
        self.output = output_neurons
        self.biased = use_bias
        self.relu = activation_function

        self.inp = torch.nn.Linear(in_features=self.input, out_features=self.hidden[0], bias=self.biased)
        self.h1 = torch.nn.Linear(in_features=self.hidden[0], out_features=self.hidden[1], bias=self.biased)
        self.h2 = torch.nn.Linear(in_features=self.hidden[1], out_features=self.hidden[2], bias=self.biased)
        self.h3 = torch.nn.Linear(in_features=self.hidden[2], out_features=self.output, bias=self.biased)
        self.out = torch.nn.Linear(in_features=self.output, out_features=self.output, bias=self.biased)
        

    def forward(self, x: torch.Tensor):
        x = self.inp(x)
        x = self.relu(x)
        x = self.h1(x)
        x = self.relu(x)
        x = self.h2(x)
        x = self.relu(x)
        x = self.h3(x)
        x = self.relu(x)
        return self.out(x)
