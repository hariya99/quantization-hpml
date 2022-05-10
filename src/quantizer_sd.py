import torch.nn as nn
import torch

class QuantizerSd:
    """
        QuantizerSd quantizes a network's weights for LSTM and Linear layers
        to support Quantization Aware Training. It does so by making a copy of 
        state_dict, changing the tensor weights with quantized/dequantized 
        weights and loading it back to the network. 
    """
    def __init__(self, scale=0.1, zero_point=10):
        self.scale = scale
        self.zero_point = zero_point 
        self.dtype = torch.qint8

    def quantize_lstm(self, lstm_layer: nn.LSTM):
        """
        """
        if(isinstance(lstm_layer, nn.LSTM)):
            # dictionary of weights
            d = lstm_layer.state_dict()
            for key in d.keys():
                d[key] = self._quantize(d[key])
            print(d)
            # d._metadata[""]["version"] = 2
            lstm_layer.load_state_dict(d)
        else:
            print("TODO: raise exception in quantize_lstm")
        
    def dequantize_lstm(self, lstm_layer: nn.LSTM):

        if(isinstance(lstm_layer, nn.LSTM)):
            # dictionary of weights
            d = lstm_layer.state_dict()
            for key in d.keys():
                d[key] = self._dequantize(d[key])
            lstm_layer.load_state_dict(d)
        else:
            print("TODO: raise exception in dequantize_lstm")        

    def _quantize(self, t: torch.tensor):
        """
        Apply quantization, convert data type to uint8 and return 
        the tensor 
        """
        t.apply_(lambda i: round((i/self.scale) + self.zero_point))
        # t.type(torch.uint8)
        t = t.to(torch.int8)
        # print(t)
        return t 
    
    def _dequantize(self, t: torch.tensor):
        """
        Convert the data type to float32 apply Dequantization and return 
        the tensor 
        """
        t = t.to(torch.float)
        t.apply_(lambda i: (i - self.zero_point) * self.scale)
        # print(t)
        return t 


if __name__ == "__main__":
    m = nn.LSTM(1, 1)
    # print("Before")
    # print(m.state_dict())
    q = QuantizerSd(0.23, 11)

    q.quantize_lstm(m)
    print("Quantization")
    print(m.state_dict())

    # q.dequantize_lstm(m)
    # print("Dequantization")
    # print(m.state_dict())




