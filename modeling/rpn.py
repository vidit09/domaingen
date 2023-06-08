# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn.functional as F

from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY, RPN_HEAD_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN, StandardRPNHead
from detectron2.structures import ImageList
from typing import  List
import time


@PROPOSAL_GENERATOR_REGISTRY.register()
class SBRPN(RPN):

    def forward(
        self,
        images,
        features,
        gt_instances= None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        val = self.rpn_head(features)

        if len(val) == 2:
            pred_objectness_logits, pred_anchor_deltas = val
            rep_clip = None
        else:
            pred_objectness_logits, pred_anchor_deltas, rep_clip = val

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits= [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]

        
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            #assert gt_instances is not None, "RPN requires gt_instances in training!"
            if gt_instances is not None:
                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
                losses = self.losses(
                    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
                )
                if rep_clip is not None :
                    if   not isinstance(rep_clip, dict):
                        ll = torch.stack(gt_labels)
                        ll = ll.reshape(ll.shape[0],-1,15)
                        valid_mask = ll.ge(0).sum(dim=-1).gt(0) # remove ignored anchors
                        ll=ll.eq(1).sum(dim=-1) # if an object is present at this location
                        ll = ll.gt(0).float()
                        
                        clip_loss = torch.nn.functional.binary_cross_entropy_with_logits(rep_clip[valid_mask],ll[valid_mask],reduction='sum')
                        losses.update({'loss_rpn_cls_clip':clip_loss/(self.batch_size_per_image*ll.shape[0])})
                    else:
                        losses.update(rep_clip)
            else:
                losses = {}
        else:   
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        # if self.training:

        # if gt_instances is None:
        out = [
                # (N, Hi*Wi*A) -> (N, Hi, Wi, A)
                score.reshape(features[ind].shape[0],features[ind].shape[-2],features[ind].shape[-1],-1)
                for ind, score in enumerate(pred_objectness_logits)
            ]
        # else:
            # b,_,h,w = features[0].shape
            # out = [1.*(torch.stack(gt_labels)==1).reshape(b,h,w,-1)]
        return out, proposals, losses
        # else:
        #     return proposals, losses




