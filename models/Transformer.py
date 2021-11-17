import torch.nn as nn
import timm


class Transformer(nn.Module):

    def __init__(self, model_name, pretrained):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.model.head = nn.Sequential() # save memory

    def forward(self, x):
        return self.model(x)


def swin_tiny(pretrained=True, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model = Transformer('swin_tiny_patch4_window7_224', pretrained=pretrained)

    return model

def deit_small(pretrained=True, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = Transformer('deit_small_patch16_224', pretrained=pretrained)

    return model
