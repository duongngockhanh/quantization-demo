import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub



# --------------------------------- Original Model ---------------------------------
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



# --------------------------------- Before Inference ---------------------------------
input = torch.rand((1, 1, 28, 28))
output = model(input)
print("-------------------Before:", output.shape)



# --------------------------------- Quantization Model ---------------------------------
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
torch.quantization.convert(quant_model, inplace=True)


# --------------------------------- Save and Load ---------------------------------
torch.save(quant_model.state_dict(), "weights/td500_resnet50_quant.pth")

quant_model2 = QuantizableSimpleModel(SimpleModel())
quant_model2.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(quant_model2, inplace=True)
torch.quantization.convert(quant_model2, inplace=True)

quant_model2.load_state_dict(torch.load("weights/td500_resnet50_quant.pth"))



# --------------------------------- After Inference ---------------------------------
input = torch.rand((1, 1, 28, 28))
output = quant_model(input)
print("-------------------After:", output.shape)

print(output.dtype) # torch.float32
