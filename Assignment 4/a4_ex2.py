import torch 

class SimpleCNN(torch.nn.Module):

    def __init__(self, 
                 input_channels: int, hidden_channels: list, 
                 use_batchnormalization: bool, num_classes: int, kernel_size: list, 
                 activation_function: torch.nn.Module = torch.nn.ReLU()):
        
        super(SimpleCNN, self).__init__()

        self.input = input_channels
        self.hidden = hidden_channels
        self.normalized = use_batchnormalization
        self.classes = num_classes
        self.kernels = kernel_size
        self.active = activation_function

        if use_batchnormalization:
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(self.input, self.hidden[0], self.kernels[0], padding=1, bias=False),
                self.active,
                torch.nn.BatchNorm2d(self.hidden[0]),
                torch.nn.Conv2d(self.hidden[0], self.hidden[1], self.kernels[1], padding=2, bias=False),
                self.active,
                torch.nn.BatchNorm2d(self.hidden[1]),
                torch.nn.Conv2d(self.hidden[1], self.hidden[2], self.kernels[2], padding=3, bias=False),
                self.active,
                torch.nn.BatchNorm2d(self.hidden[2]),
                torch.nn.Flatten(),
                torch.nn.Linear(896000, self.classes)
            )
        else: 
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(self.input, self.hidden[0], self.kernels[0], padding=1),
                self.active,
                torch.nn.Conv2d(self.hidden[0], self.hidden[1], self.kernels[1], padding=2),
                self.active,
                torch.nn.Conv2d(self.hidden[1], self.hidden[2], self.kernels[2], padding=3),
                self.active,
                torch.nn.Flatten(),
                torch.nn.Linear(896000, self.classes)
            )

    def forward(self, input_images: torch.Tensor):
        x = input_images
        return self.model(x)
    