"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase
from onmt.decoders.transformer import TransformerDecoder
#from onmt.decoders.cnn_decoder import CNNDecoder

"""
str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
"""
str2dec = {"transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "str2dec"]