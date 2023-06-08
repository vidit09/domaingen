from pydoc import classname
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


import torchvision.transforms as T


from detectron2.layers import ShapeSpec
from detectron2.data import MetadataCatalog

from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY,  Res5ROIHeads
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from .box_predictor import ClipFastRCNNOutputLayers

def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


@ROI_HEADS_REGISTRY.register()
class ClipRes5ROIHeads(Res5ROIHeads):   
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)
        clsnames = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes").copy()

        # import pdb;pdb.set_trace()
        ##change the labels to represent the objects correctly
        for name in  cfg.MODEL.RENAME:
            ind = clsnames.index(name[0])
            clsnames[ind] = name[1]
       
        out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3) ### copied 
        self.box_predictor = ClipFastRCNNOutputLayers(cfg, ShapeSpec(channels=out_channels, height=1, width=1), clsnames)
        self.clip_im_predictor = self.box_predictor.cls_score # should call it properly
        self.device = cfg.MODEL.DEVICE
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        crops: Optional[List[Tuple]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            # import pdb;pdb.set_trace()
            loss_crop_im = None
            if crops is not None:
                crop_im = list()#[x[0] for x in crops] #bxcropx3x224x224
                crop_boxes = list()#[x[1].to(self.device) for x in crops] #bxcropsx4
                keep = torch.ones(len(crops)).bool()
                
                for ind,x in enumerate(crops):
                    if len(x) == 0:
                        keep[ind] = False   
                        continue
                    crop_im.append(x[0])
                    crop_boxes.append(x[1].to(self.device))
                    
                c = self._shared_roi_transform(
                            [features[f][keep] for f in self.in_features], crop_boxes) #(b*crops)x2048x7x7
                loss_crop_im, _ = self.clip_im_predictor.forward_crops(crop_im,crops_features.mean(dim=[2, 3]))

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        # import pdb;pdb.set_trace()
        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

            if loss_crop_im is not None:
                losses.update(loss_crop_im)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class ClipRes5ROIHeadsAttn(ClipRes5ROIHeads): 
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)
        # self.res5 = None
    
    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.fwdres5(x)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        crops: Optional[List[Tuple]] = None,
        backbone = None
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        self.fwdres5 = backbone.forward_res5

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            # import pdb;pdb.set_trace()
            loss_crop_im = None
            if crops is not None:
                crop_im = list()#[x[0] for x in crops] #bxcropx3x224x224
                crop_boxes = list()#[x[1].to(self.device) for x in crops] #bxcropsx4
                keep = torch.ones(len(crops)).bool()
                
                for ind,x in enumerate(crops):
                    if len(x) == 0:
                        keep[ind] = False   
                        continue
                    crop_im.append(x[0])
                    crop_boxes.append(x[1].to(self.device))
                    
                crops_features = self._shared_roi_transform(
                            [features[f][keep] for f in self.in_features], crop_boxes) #(b*crops)x2048x7x7
                crops_features = backbone.attention_global_pool(crops_features)
                loss_crop_im, _ = self.clip_im_predictor.forward_crops(crop_im,crops_features)

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        attn_feat = backbone.attention_global_pool(box_features)
        predictions = self.box_predictor([attn_feat,box_features.mean(dim=(2,3))])
        # import pdb;pdb.set_trace()
        if self.training:
            del features
            
            losses = self.box_predictor.losses(predictions, proposals)
            # if torch.isnan(losses['loss_cls']):
            #     import pdb;pdb.set_trace()
           
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

            if loss_crop_im is not None:
                losses.update(loss_crop_im)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


