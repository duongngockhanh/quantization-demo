import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32*28*28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel().to(device)
model = model.eval().to(torch.device('cpu'))

input = torch.rand((1, 1, 28, 28))
output = model(input)
print("-------------------Before:", output.shape)

class QuantizableSimpleModel(nn.Module):
    def __init__(self, model):
        super(QuantizableSimpleModel, self).__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

quant_model = QuantizableSimpleModel(model)
quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(quant_model, inplace=True)
quant_model = torch.quantization.convert(quant_model, inplace=True)

input = torch.rand((1, 1, 28, 28))
output = quant_model(input)
print("-------------------After:", output.shape)