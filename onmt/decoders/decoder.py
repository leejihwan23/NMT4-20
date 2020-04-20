import torch.nn as nn


class DecoderBase(nn.Module):

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings):

        raise NotImplementedError

