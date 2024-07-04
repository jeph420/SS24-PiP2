import torch
import torch.nn as nn


class SimpleNetwork(nn.Module):

    def __init__(
            self,
            input_neurons: int,
            hidden_neurons: list,
            output_neurons: int,
            use_bias: bool,
            activation_function: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.use_bias = use_bias
        self.activation_function = activation_function

        self.input_layer = nn.Linear(self.input_neurons, self.hidden_neurons[0], bias=self.use_bias)
        self.hidden_layer_1 = nn.Linear(self.hidden_neurons[0], self.hidden_neurons[1], bias=self.use_bias)
        self.hidden_layer_2 = nn.Linear(self.hidden_neurons[1], self.hidden_neurons[2], bias=self.use_bias)
        # hidden layer 3 + the output layer
        self.output_layer = nn.Linear(self.hidden_neurons[2], self.output_neurons, bias=self.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_function(self.input_layer(x))
        x = self.activation_function(self.hidden_layer_1(x))
        x = self.activation_function(self.hidden_layer_2(x))
        x = self.output_layer(x)
        return x

#####------------ Alternative method using ModuleList ------------#####
# class SimpleNetwork(nn.Module):
#
#     def __init__(
#             self,
#             input_neurons: int,
#             hidden_neurons: list,
#             output_neurons: int,
#             use_bias: bool,
#             activation_function: nn.Module = nn.ReLU()
#     ):
#         super().__init__()
#         self.input_neurons = input_neurons
#         self.hidden_neurons = hidden_neurons
#         self.output_neurons = output_neurons
#         self.use_bias = use_bias
#         self.activation_function = activation_function
#
#         self.input_layer = nn.Linear(self.input_neurons, self.hidden_neurons[0], bias=self.use_bias)
#
#         self.layers = nn.ModuleList([])
#         self.layers.append(
#             nn.Sequential(
#                 nn.Linear(self.hidden_neurons[0], self.hidden_neurons[1], bias=self.use_bias),
#                 nn.ReLU()
#             ))
#         for i in range(1,len(self.hidden_neurons)-1):
#             self.layers.append(
#                 nn.Sequential(
#                     nn.Linear(self.hidden_neurons[i], self.hidden_neurons[i+1], bias=self.use_bias),
#                     nn.ReLU()
#                 ) )
#
#         self.output_layer = nn.Linear(self.hidden_neurons[-1], self.output_neurons, bias=self.use_bias)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.activation_function(self.input_layer(x))
#
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#
#         x = self.output_layer(x)
#         return x


#####------------ Alternative method using Sequential ------------#####
# class SimpleNetwork(nn.Module):
#
#     def __init__(
#             self,
#             input_neurons: int,
#             hidden_neurons: list,
#             output_neurons: int,
#             use_bias: bool,
#             activation_function: nn.Module = nn.ReLU()
#     ):
#         super().__init__()
#         self.input_neurons = input_neurons
#         self.hidden_neurons = hidden_neurons
#         self.output_neurons = output_neurons
#         self.use_bias = use_bias
#         self.activation_function = activation_function
#
#         self.input_layer = nn.Linear(self.input_neurons, self.hidden_neurons[0], bias=self.use_bias)
#
#         layers = []
#         layers.append(nn.Linear(self.hidden_neurons[0], self.hidden_neurons[1], bias=self.use_bias))
#         layers.append(nn.ReLU())
#         for i in range(1, len(self.hidden_neurons)-1):
#             layers.append( nn.Linear(self.hidden_neurons[i], self.hidden_neurons[i+1], bias=self.use_bias) )
#             layers.append(nn.ReLU())
#         self.layers = nn.Sequential(*layers)
#
#         self.output_layer = nn.Linear(self.hidden_neurons[-1], self.output_neurons, bias=self.use_bias)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.activation_function(self.input_layer(x))
#         x = self.layers(x)
#         x = self.output_layer(x)
#         return x




if __name__ == "__main__":
    torch.random.manual_seed(1234)
    simple_network = SimpleNetwork(40, [10, 20, 30], 5, True)
    # print(simple_network)
    input = torch.randn(1, 40, requires_grad = False)
    output = simple_network(input)
    print(output)
