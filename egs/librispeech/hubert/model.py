# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy

#from foundnet.models.sequence import SequenceBatch
#from foundnet.models.wav2vec2.vector_quantizer import (
#    VectorQuantizer,
#    VectorQuantizerOutput,
#)
from foundnet.nn.ops import repeat_interleave
from foundnet.nn.transformer.projection import Linear
from foundnet.nn.transformer import TransformerEncoder
from foundnet.utils.module import check_model_dim
from foundnet.utils.typing import DataType, Device
 
from frontend import Wav2Vec2Frontend
from masker import Wav2Vec2Masker, apply_temporal_mask

@dataclass
class SequenceBatch:
    """Represents a sequence batch."""

    seqs: Tensor
    """The sequences. *Shape:* :math:`(N,S,*)`, where :math:`N` is the batch
    size, :math:`S` is the sequence length, and :math:`*` is any number of
    sequence-specific dimensions including none."""

    seq_lens: Optional[Tensor]
    """An array where each element represents the length of the sequence at the
    same index in :attr:`seqs`. *Shape:* :math:`(N)`, where :math:`N` is the
    batch size."""

    example: Any = None
    """The data example from which this batch was constructed."""

    @property
    def batch_size(self) -> int:
        """The size of the batch."""
        return self.seqs.size(0)

    def num_tokens(self) -> Tensor:
        """Return the number of tokens."""
        if self.seq_lens is None:
            return torch.full((), self.seqs.numel(), device=self.seqs.device)

        return self.seq_lens.sum()



class HubertModel(Module):
    """Represents a Hubert model as described in
    :cite:t:`https://arxiv.org/pdf/2106.07447.pdf`."""

    model_dim: int
    encoder_frontend: Wav2Vec2Frontend
    encoder: TransformerEncoder
    masker: Wav2Vec2Masker
    final_proj: Linear
    final_target_proj: Linear
    logit_temp: float
    sim_type: str
    predict_layers: str 
    pred_masked_weight:  float
    pred_nomasked_weight: float 
    loss_weights: int
     
    def __init__(
        self,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        masker: Wav2Vec2Masker,
        final_dim: int,
        *,
        final_proj_bias: bool = True,
        logit_temp: float = 0.1,
        sim_type: str = "cosine",
        predict_layers:  str = "[12]",
        pred_masked_weight:  float = 1.0,
        pred_nomasked_weight: float = 0.0,
        loss_weights: float = 10.0
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder (i.e. context network).
        :param masker:
            The temporal/spatial feature masker.
        :param final_dim:
            The dimensionality of the final projection that is applied to
            context network outputs and quantized targets.
        :param final_proj_bias:
            If ``True``, the final projection learns an additive bias.
        :param logit_temp:
            The temperature to divide logits by.
        """
        super().__init__()

        #model_dim = encoder.model_dim

        self.model_dim = encoder.model_dim

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder

        self.masker = masker


        #self.final_proj = Linear(
        #    model_dim, final_dim, final_proj_bias, device=device, dtype=dtype
        #)
        ##  for feat postprocessing
        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = Linear(
                self.model_dim, final_dim * len(dictionaries), final_proj_bias,device=device, dtype=dtype
            )
        else:
            self.final_proj = Linear(self.model_dim, final_dim,final-proj_bias,device=device, dtype=dtype)
        ## for label postprocessing
        self.num_classes = [len(d) for d in dictionaries]
        self.label_embs_concat = nn.Parameter(
            torch.FloatTensor(sum(self.num_classes), final_dim)
        )
        torch.nn.init.uniform_(self.label_embs_concat)

        #self.final_target_proj = Linear(
        #    self.quantizer.output_dim,
        #    final_dim,
        #    bias=True,
        #    device=device,
        #    dtype=dtype,
        #)

        if self.separate_label_embeds:
            if self.separate_layer_targets or not self.untie_final_proj:
                self.final_proj = torch.nn.Sequential(
                    *[
                        nn.Linear(cfg.encoder_embed_dim, cfg.final_dim)
                        for _ in range(len(self.predict_layers))
                    ]
                )
            else:
                self.final_proj = torch.nn.Sequential(
                    *[
                        nn.Linear(
                            cfg.encoder_embed_dim, cfg.final_dim * len(dictionaries)
                        )
                        for _ in range(len(self.predict_layers))
                    ]
                )
        else:
            if self.separate_layer_targets or not self.untie_final_proj:
                self.final_proj = nn.Linear(cfg.encoder_embed_dim, cfg.final_dim)
            else:
                self.final_proj = nn.Linear(
                    cfg.encoder_embed_dim, cfg.final_dim * len(dictionaries)
                )

        self.logit_temp = logit_temp
        self.sim_type = sim_type
        self.predict_layers = predict_layers
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomasked_weight = pred_nomasked_weight
        self.loss_weights = loss_weights 

        check_model_dim(self)

    def forward(self, batch: SequenceBatch) -> "Wav2Vec2Output":
        """
        :param batch:
            The batch of sequences to process.
        """
        seqs, padding_mask, temporal_mask,features_pen = self.run_frontend(
            batch.seqs, batch.seq_lens
        )

        # TODO: Should pad for fp16?
        encoder_output, layer_results, _ = self.encoder(seqs, padding_mask, layer=self.predict_layers)

        #return self.quantize_and_contrast(encoder_output, targets, temporal_mask)

    def run_frontend(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """Run the encoder frontend in pretraining mode.

        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``seqs``. *Shape:* :math:`(N)`, where :math:`N` is
            the batch size.

        :returns:
            - The processed sequences to pass to the Transformer encoder.
              *Shape:* :math:`(N,S_{out},M)`, where :math:`N` is the batch size,
              :math:`S_{out}` is the output sequence length, and :math:`M` is
              the dimensionality of the model.
            - The float padding mask of the processed sequences. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
            - The non-quantized context network targets that have been extracted
              from the input sequences. *Shape:* :math:`(N,S_{msk},M)`, where
              :math:`N` is the batch size, :math:`S_{msk}` is the masked
              sequence length, and :math:`M` is the dimensionality of the model.
            - The boolean temporal mask that has been applied to extract the
              context network targets. *Shape:* :math:`(N,S_{out})`, where
              :math:`N` is the batch size and :math`S_{out}` is the output
              sequence length.
        """
        frontend = self.encoder_frontend

        seqs, seq_lens = frontend.extract_features(seqs, seq_lens)
   
        features_pen = seq.float().pow(2).mean()
        # We use the extracted features as context network targets after masking
        # and quantization.
        #targets = seqs.clone().detach()

        if frontend.first_pass_dropout is not None:
            targets = frontend.first_pass_dropout(targets)

        seqs, padding_mask, temporal_mask = frontend.process_features(
            seqs, seq_lens, self.masker
        )

        assert temporal_mask is not None

        #targets = apply_temporal_mask(targets, temporal_mask)

        return seqs, padding_mask,temporal_mask,features_pen

    def post_feat_and_target(self, layerr_results,):
        logit_m_list = []
        logit_u_list = []
        target_m_list = []
        target_u_list = []
        if self.separate_layer_targets:
            assert len(layer_results) == len(self.final_proj)
            assert len(layer_results) == len(self.label_embs_concat)

        for i, layer_x in enumerate(
            layer_results
        ):  # , final_proj, label_embs in zip(layer_results, self.final_proj, label_embs_concat):
            if self.separate_label_embeds:
                final_proj = self.final_proj[i]
            else:
                final_proj = self.final_proj

            if self.separate_label_embeds or self.separate_layer_targets:
                label_embs = self.label_embs_concat[i]
            else:
                label_embs = self.label_embs_concat[0]

            if not self.separate_layer_targets:
                label_embs_list = label_embs.split(self.num_classes, 0)
            else:
                label_embs_list = [label_embs[: self.num_classes[i]]]                   

            proj_x = final_proj(layer_x)
            if self.untie_final_proj:
                proj_x_list = proj_x.chunk(
                    len(self.num_classes), dim=-1
                )  # len(proj_x_list) = len(self.num_classes)
            else:
                proj_x_list = [proj_x for _ in self.num_classes]
            logit_list = [
                self.compute_logits(proj, emb).view(-1, num_class)
                for proj, emb, num_class in zip(
                    proj_x_list, label_embs_list, self.num_classes
                )
            ]  # [[B*T, V]]
            if not self.skip_masked:
                mask = torch.logical_and(mask_indices, ~padding_mask).view(-1)  # [B*T]
                logit_m_list += [logit[mask] for logit in logit_list]
                target_m_list += [target.view(-1)[mask].long() for target in target_list]
            else:
                logit_m_list += [None for _ in target_list]

            if not self.skip_nomask:
                unmask = torch.logical_and(~mask_indices, ~padding_mask).view(-1)  # [B*T]
                logit_u_list += [logit[unmask] for logit in logit_list]
                target_u_list += [target.view(-1)[unmask].long() for target in target_list]
            else:
                logit_u_list += [None for _ in target_list]

        net_output = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "features_pen": features_pen,
        }

        return  HubertOuput(
                net_output,
                self.pred_masked_weight,
                self.pred_nomasked_weight,
                self.loss_weights,
            )       


    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == "dot":
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == "cosine":
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(
                dim=-1
            )  # [B*T, V]
            denom = (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (
                emb_mat**2
            ).sum(dim=-1).sqrt().unsqueeze(
                dim=0
            )  # [B*T, V]
            logits = (nom / denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits
    def extra_repr(self) -> str:
        """:meta private:"""
        return (
            f"model_dim={self.model_dim}, "
            f"logit_temp={self.logit_temp}, "

        )


@dataclass
class HubertOutput:
    """Holds the output of a Hubert model."""
    net_output: Dict
    pred_masked_weight: float   
    """ weight for predictive loss for masked frames"""
    pred_nomask_weight: float
    """ weight for predictive loss for unmasked frames"""
    loss_weights: Optional[List[float]] 
    """ weights for additional loss terms""" 

    def compute_loss(self) -> "HubertLoss":
        """Compute the loss."""
        ssl_loss, sample_size = self.compute_ssl_loss()

        extra_loss = self.compute_extra_loss()
        extra_loss = extra_loss.float() * sample_size

        loss = ssl_loss + self.loss_weights * extra_loss

        return HubertLoss(loss, ssl_loss, extra_loss)

    def compute_ssl_loss(self) -> Tensor:
        """Compute the ssl loss."""
        loss_m_list = []
        logp_m_list = self.get_logits(net_output, True)
        targ_m_list = self.get_targets(net_output, True)
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = cross_entropy(logp_m, targ_m, reduction="sum")
            loss_m_list.append(loss_m)
            #logging_output[f"loss_m_{i}"] = loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += targ_m_list[0].numel()

        loss_u_list = []
        logp_u_list = self.get_logits(net_output, False)
        targ_u_list = self.get_targets(net_output, False)
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = cross_entropy(logp_u, targ_u, reduction="sum")
            loss_u_list.append(loss_u)
            #logging_output[f"loss_u_{i}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += targ_u_list[0].numel()
        return loss,sample_size
                    
    def compute_extra_loss(self):
       
        return net_output["features_pen"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

@dataclass
class HubertLoss:
    """Holds the loss of a Hubert model."""

    total: Tensor
    """The weighted total loss."""

    ssl_loss: Tensor
    """The hubert ssl loss loss."""

    extra_loss: Tensor
    """The audiofeature penalty loss before transformer encoder."""

    def backward(self) -> None:
        """Compute the gradient of the loss."""
        self.total.backward()
