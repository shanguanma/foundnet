# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax
from foundnet.nn.functional import nll_loss

from foundnet.models.transformer.frontend import TransformerFrontend
from foundnet.nn.transformer.incremental_state import IncrementalStateBag
from foundnet.nn.transformer.projection import Projection
from foundnet.nn.transformer import TransformerDecoder
from foundnet.utils.module import check_model_dim
from foundnet.utils.typing import finaloverride


@final
class TransformerDecoderModel(Module):
    """Represents a Transformer-based decoder model."""

    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    target_pad_idx: Optional[int]

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        target_pad_idx: Optional[int],
    ) -> None:
        """
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The decoder.
        :param final_proj:
            The projection to apply to decoder outputs to produce logits.
        :param target_pad_idx:
            The index of the pad symbol in the target vocabulary.
        """
        #model_dim = decoder.model_dim

        #super().__init__(model_dim)
        super().__init__()
        self.model_dim = decoder.model_dim
      
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.target_pad_idx = target_pad_idx

        check_model_dim(self)

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Decode the specified sequences.

        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder output. *Shape:* :math:`(N,S,M)`, where :math:`N` is
              the batch size, :math:`S` is the target sequence length, and
              :math:`M` is the dimensionality of the model.
            - The float padding mask of the decoder output. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math:`S` is
              the target sequence length.
        """
        seqs, padding_mask = self.decoder_frontend(seqs, seq_lens, state_bag=state_bag)

        decoder_output, decoder_padding_mask = self.decoder(
            seqs, padding_mask, state_bag=state_bag
        )

        return decoder_output, decoder_padding_mask

    @finaloverride
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        """Produce logits for next-step prediction.

        :param decoder_output:
            The decoder output. *Shape:* :math:`(N,S,M)`, where :math:`N` is the
            batch size, :math:`S` is the sequence length, and :math:`M` is the
            dimensionality of the model.
        :param decoder_padding_mask:
            The float padding mask of the decoder output. *Shape:*
            :math:`(N,S)`, where :math:`N` is the batch size and :math:`S` is
            the sequence length.
        """

        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.target_pad_idx)

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@dataclass
class SequenceModelOutput:
    """Holds the output of a sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the sequence length, and :math:`T`
    is the size of the target vocabulary."""

    pad_idx: Optional[int] = None
    """The index of the pad symbol in the target vocabulary."""

    def compute_loss(
        self,
        targets: Tensor,
        *,
        ignore_prefix_size: int = 0,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        """Compute the negative log-likelihood loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the sequence length.
        :param ignore_prefix_size:
            The number of logits from the beginning of the sequence that should
            be ignored in the loss computation.
        :param label_smoothing:
            The amount of label smoothing when computing the loss.
        """
        if ignore_prefix_size > 0:
            logits = self.logits[:, ignore_prefix_size:, :]
        else:
            logits = self.logits

        if ignore_prefix_size > 0:
            targets = targets[:, ignore_prefix_size:]

        # For numerical stability run in single precision.
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        return nll_loss(lprobs, targets, self.pad_idx, label_smoothing=label_smoothing)
