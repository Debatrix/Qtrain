import torch
import torch.nn as nn

from src.arch.module import Attention_pooling, Resnet18_encoder, UNet_encoder, UNet_decoder, AttUNet_encoder, AttUNet_decoder, PredictHead


# IQA
class ResUnet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResUnet, self).__init__()
        self.encoder = Resnet18_encoder(pretrained)
        self.decoder = AttUNet_decoder()
        self.attpooling = Attention_pooling()
        self.predictor = PredictHead(2, 512)

    def forward(self, input):
        feature = self.encoder(input)
        heatmap = self.decoder(*feature)
        prediction = self.attpooling(feature[-1], heatmap)
        prediction = self.predictor(prediction)
        return {'heatmap': heatmap, 'prediction': prediction}


# class AttUnet(nn.Module):
#     def __init__(self):
#         super(AttUnet, self).__init__()
#         self.encoder = AttUNet_encoder()
#         self.decoder = AttUNet_decoder()
#         self.attpooling = Attention_pooling()
#         self.predictor = PredictHead(2, 512)

#     def forward(self, input):
#         feature = self.encoder(input)
#         heatmap = self.decoder(*feature)
#         prediction = self.attpooling(feature[-1], heatmap)
#         prediction = self.predictor(prediction)
#         return {'heatmap': heatmap, 'prediction': prediction}


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder = UNet_encoder()
        self.decoder = UNet_decoder()
        self.attpooling = Attention_pooling()
        self.predictor = PredictHead(2, 64)

    def forward(self, input):
        feature = self.encoder(input)
        heatmap = self.decoder(*feature)
        prediction = self.attpooling(feature[-1], heatmap)
        prediction = self.predictor(prediction)
        return {'heatmap': heatmap, 'prediction': prediction}


if __name__ == "__main__":
    m = Unet()
    input = torch.rand((3, 1, 512, 645))
    output = m(input)
    print(output['heatmap'].shape, output['prediction'].shape)
