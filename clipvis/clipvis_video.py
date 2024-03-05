# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

from .modeling.criterion_video import SetCriterion_video
from .modeling.matcher_video import HungarianMatcher_video

from scipy.optimize import linear_sum_assignment

from .modeling.transformer_decoder.mask2former_transformer_decoder import MaskPooling, get_classification_logits

import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
import einops

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]


@META_ARCH_REGISTRY.register()
class CLIPVIS_Video(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,

        train_metadata,
        test_metadata,

        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        num_frames: int,
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    
        num_t:int,
        num_k:int,
        window_inference: bool,
        test_num_classes: int
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold

        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.num_frames = num_frames

        self.window_inference = window_inference

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.num_t=num_t
        self.num_k=num_k
        self.test_num_classes=test_num_classes

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference


        self.mask_pooling = MaskPooling()

        self.train_text_classifier = None
        self.test_text_classifier = None

        self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(test_metadata, train_metadata)

    def prepare_class_names_from_metadata(self, metadata, train_metadata):

        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
        except:
            class_names = split_labels(metadata.thing_classes)
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
        
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        return num_templates, class_names
    
    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:

                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        object_weight = cfg.MODEL.MASK_FORMER.OBJECT_WEIGHT
        maskiou_weight = cfg.MODEL.MASK_FORMER.MASKIOU_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher_video(
            cost_object=object_weight,
            cost_iou=maskiou_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        weight_dict = {"loss_object_ce": object_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,"loss_iou":maskiou_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion_video(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,

            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),

            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,

            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "num_t":cfg.TEST.NUM_FRAME,
            "num_k":cfg.TEST.NUM_TOPK,
            "window_inference": cfg.TEST.WINDOW_INFERENCE,
            "test_num_classes": cfg.TEST.TEST_NUM_CLASSES
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [x.to(self.device) for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        text_classifier, num_templates = self.get_text_classifier()

        if not self.training and self.window_inference:
            outputs,features_list = self.run_window_inference(images.tensor)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

        if self.training:

            if "instances" in batched_inputs[0]:
                targets = self.prepare_targets(batched_inputs, images)
            else:
                targets = None

            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs = self.post_processing(outputs)

            pred_object_scores = outputs["pred_obj_logits"]
            pred_per_frame_object_scores = outputs["pred_per_frame_obj_logits"]
            pred_masks = outputs["pred_masks"]
            pred_maskious=outputs['pred_ious']

            dense_feature=features_list["clip_vis_dense"]
            cls_scores_list=[]
            for i in range (pred_masks.shape[2]):
                mask_for_pooling = F.interpolate(pred_masks[:,:,i,:,:], size=dense_feature.shape[-2:],mode='bilinear', align_corners=False)
                if "convnext" in self.backbone.model_name.lower():
                    pooled_clip_feature_dense = self.mask_pooling(dense_feature[i,:,:,:][None,:], mask_for_pooling)                 
                    pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature_dense)
                elif "rn" in self.backbone.model_name.lower():
                    pooled_clip_feature = self.backbone.visual_prediction_forward(dense_feature[i,:,:,:][None,:], mask_for_pooling)
                else:
                    raise NotImplementedError
                cls_score = get_classification_logits(pooled_clip_feature, text_classifier, self.backbone.clip_model.logit_scale, num_templates)
                if "rn" in self.backbone.model_name.lower():
                    valid_masking = ((mask_for_pooling > 0).to(mask_for_pooling).sum(-1).sum(-1))> 0
                    valid_masking = ~valid_masking
                    cls_score[valid_masking]=0.0
                cls_scores_list.append(cls_score.unsqueeze(2))
            per_frame_cls_scores=torch.cat(cls_scores_list,dim=2)

            per_cls_results=per_frame_cls_scores.softmax(-1)

            pred_maskious_score=(per_cls_results*pred_maskious[0])**0.5
            pred_maskious_score=pred_maskious_score.mean(2)

            object_scores = F.softmax(pred_object_scores, dim=-1)[..., :-1]
           
            final_cls_score = (pred_maskious_score * object_scores) ** 0.5

            del outputs
            processed_results = []
            for mask_cls_result, mask_object_cls_result, mask_pred_result, input_per_image, image_size in zip(
                final_cls_score, pred_object_scores, pred_masks, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_object_cls_result,\
                        mask_pred_result, image_size,\
                        per_cls_results[0], pred_per_frame_object_scores[0])
                processed_results[-1]["instances"] = instance_r
                processed_results[-1]["interpolate_meta"] = (image_size, height, width, (images.tensor.shape[-2], images.tensor.shape[-1]))

            return processed_results
    
    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q h w -> b q () h w')

        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q h w -> b q () h w'
                )
        gt_instances = []
        for targets_per_video in targets:
            # labels: N (num instances)
            # ids: N, num_labeled_frames
            # masks: N, num_labeled_frames, H, W
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels'][:,f]
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})
        return outputs, gt_instances

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            obj_label=[_num_instance, self.num_frames]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            gt_obj_per_video = torch.zeros(obj_label, dtype=torch.long, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size
                per_valid_idx = (targets_per_frame.gt_ids == -1)
                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                
                gt_obj_per_video[per_valid_idx,f_i]=1
                
            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = torch.zeros_like(targets_per_frame.gt_classes)  
            gt_classes_per_video=gt_classes_per_video[valid_idx]        # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_obj_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def instance_inference(self, mask_cls, mask_object_cls, mask_pred, image_size, mask_per_frame_cls, mask_per_frame_object_cls):

        scores = mask_cls
        mask_per_frame_object_scores = (mask_per_frame_cls.sigmoid() * mask_per_frame_object_cls.softmax(-1)[...,:-1]) ** 0.5
        labels = torch.arange(self.test_num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.test_num_classes, rounding_mode='trunc')
        mask_pred = mask_pred[topk_indices]

        mask_per_frame_object_scores = mask_per_frame_object_scores[topk_indices]
        Q = mask_per_frame_object_scores.shape[0]
        mask_per_frame_object_scores = mask_per_frame_object_scores[torch.arange(Q),:,labels_per_image]

        mask_ignore = mask_per_frame_object_scores < scores_per_image[:,None] * 0.1
        mask_pred[mask_ignore] = -1

        result = Instances(image_size)
        result.pred_masks = mask_pred
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * (result.pred_masks > 0).float().flatten(1)).sum(1) / ((result.pred_masks > 0).float().flatten(1).sum(1) + 1e-6)
        result.scores = (scores_per_image * mask_scores_per_image)
        result.pred_classes = labels_per_image
        return result
    
    def match_from_embds(self, tgt_embds, cur_embds):

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))
        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices
    def match_from_vote_embds(self, tgt_embds, cur_embds):
            cos_score_list=[]
            cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
            indices_lists=[]
            cost_list=[]
            T = len(tgt_embds)
            top_t=self.num_k
            for i in range(len(tgt_embds)):
                tgt_embd = tgt_embds[i] / tgt_embds[i].norm(dim=1)[:, None]
                cos_sim = torch.mm(cur_embds, tgt_embd.transpose(0, 1))
                cost_embd = 1 - cos_sim
                C = 1.0 * cost_embd
                C = C.cpu()
                cos_score=C*(i+1)
                cos_score_list.append(cos_score)
                indice = linear_sum_assignment(C.transpose(0, 1))  
                indices_lists.append(indice[1]) 
                cost_score=cost_embd[indice[1],indice[0]].sum()
                cost_list.append(cost_score)
            costs=torch.stack(cost_list,dim=0)
            cos_scores=torch.stack(cos_score_list,dim=0)
            
            if costs.shape[0]<top_t:
                min_cost_indice=torch.topk(costs,k=1,largest=False)
                indices=indices_lists[min_cost_indice[1]]
            else:
                min_cost_indice=torch.topk(costs,k=top_t,largest=False)
                c_score=cos_scores[min_cost_indice[1]].mean(0)
                final_indices = linear_sum_assignment(c_score.transpose(0, 1)) 
                indices=final_indices[1]
            return indices

    def post_processing(self, outputs):
        pred_obj_logits, pred_masks, pred_embds,pred_ious =  outputs['pred_object_logits'], outputs['pred_masks'], outputs['pred_embds'],outputs['pred_ious']

        pred_obj_logits = list(torch.unbind(pred_obj_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))
        pred_ious =list(torch.unbind(pred_ious))

        out_obj_logits = []
        out_masks = []
        out_ious=[]
       
        out_obj_logits.append(pred_obj_logits[0])
        out_masks.append(pred_masks[0])
        out_ious.append(pred_ious[0])
       
        embed_memory=deque(maxlen=self.num_t)
        embed_memory.append(pred_embds[0])

        for i in range(1, len(pred_embds)):
            indices = self.match_from_vote_embds(embed_memory,pred_embds[i])

            embed_memory.append(pred_embds[i][indices, :])
            out_obj_logits.append(pred_obj_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_ious.append(pred_ious[i][indices,:])
           
        per_frame_out_obj_logits = torch.stack(out_obj_logits, dim=1)
        per_frame_out_ious=torch.stack(out_ious,dim=1)

        out_obj_logits = sum(out_obj_logits)/len(out_obj_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_obj_logits = out_obj_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)
        per_frame_out_ious=per_frame_out_ious.unsqueeze(0)
        per_frame_out_obj_logits=per_frame_out_obj_logits.unsqueeze(0)

        outputs['pred_obj_logits'] = out_obj_logits
        outputs['pred_per_frame_obj_logits'] = per_frame_out_obj_logits
        outputs['pred_masks'] = out_masks
        outputs['pred_ious']=per_frame_out_ious
   
        return outputs

    def run_window_inference(self, images_tensor, window_size=10):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        features_dense_list=[]

        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks']
            out_list.append(out)
            features_dense_list.append(features["clip_vis_dense"])
            del features

        # merge outputs
        outputs = {}
       
        outputs['pred_object_logits'] = torch.cat([x['pred_object_logits'] for x in out_list], dim=0).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=0).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=0).detach()
        outputs['pred_ious'] = torch.cat([x['pred_ious'] for x in out_list], dim=0).detach()
        features_list={}
        features_list["clip_vis_dense"]=torch.cat(features_dense_list,dim=0)
        return outputs,features_list

def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, :, :img_size[0], : img_size[1]]
    # interpolate in a loop to avoid OOM
    window_size = 20
    sliced_result = []
    for i in range(0, result.shape[1], window_size):
        sliced_result.append(F.interpolate(
            result[:,i:i+window_size], size=(output_height, output_width), mode="bilinear", align_corners=False
        ))
    result = torch.cat(sliced_result, dim=1)
    return result

