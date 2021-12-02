import torch

# define a floating point model where some layers could be statically quantized
class QuantizedModel(torch.nn.Module):
    def __init__(self, subnet):
        super(QuantizedModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.subnet = subnet
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.quant(x)
        x = self.subnet(x)
        x = self.dequant(x)
        return x



 


