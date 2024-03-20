# CLIP-VIS: Adapting CLIP for Open-Vocabulary Video Instance Segmentation

This repo is the official implementation of `CLIP-VIS: Adapting CLIP for Open-Vocabulary Video Instance Segmentation`


<div style="display: flex;">
    <img width="18%" src="assert/00077.gif">
    <img width="18%" src="assert/00251.gif">
    <img width="18%" src="assert/00304.gif">
    <img width="18%" src="assert/00290.gif">
    <img width="18%" src="assert/00139.gif">
</div>
<div style="display: flex;">
    <img width="18%" src="assert/00129.gif" >
    <img width="18%" src="assert/00134.gif">
    <img width="18%" src="assert/00181.gif">
    <img width="18%" src="assert/00202.gif">
    <img width="18%" src="assert/00254.gif">
</div>
<div style="display: flex;">
    <img width="46%" src="assert/2d802cb8.gif">
    <img width="46%" src="assert/2112a80d.gif">
</div>
<div style="display: flex;">
    <img width="46%" src="assert/86b8e4ec.gif">
    <img width="46%" src="assert/30446667.gif">
</div>


## Introduction
<img width="100%" src="assert/overview.png"><br>
- We present a simple encoder-decoder to adapt CLIP
for open-vocabulary video instance segmentation, called
CLIP-VIS. Based on frozen CLIP, our CLIP-VIS retains
strong the zero-shot classification ability to various instance categories.
- We design a temporal topK-enhanced matching strategy,
which adaptively selects K mostly matching frames to
perform query matching.
- We further introduce a weighted open-vocabulary classification module, which refines mask classification by
correlating mask prediction and classification.
- Our CLIP-VIS achieves superior performance on multiple
datasets. On validation of LV-VIS dataset, our CLIP-VIS
outperforms OV2Seg by 4.6% and 11.0% AP using the
backbone ResNet50 and ConNext-B. When evaluating
novel categories, our CLIP-VIS outperforms OV2Seg by
10.9% and 24.0% AP, which demonstrates the good
zero-shot classification ability to novel categories. When
using the ResNet50 as backbone, our CLIP-VIS outperforms OpenVIS by 1.7% AP on the validation set of
BURST dataset.

For further details, please check out our [paper](http://arxiv.org/abs/2403.12455).
## Installation
Please follow [installation](INSTALL.md). 

## Data Preparation
Please follow [dataset preperation](datasets/README.md).

## Training
We provide shell scripts for training on image datasets and video datasets. ```scripts/train.sh``` trains the model on LVIS or COCO dataset. ```scripts/train_video.sh``` fine-tune the model on YTVIS2019 dataset.

To train or evaluate the model in different environments, modify the given shell script and config files accordingly.

### Training script
```bash
sh scripts/train.sh [CONFIG] [NUM_GPUS] [BATCH_SIZE] [OUTPUT_DIR] [OPTS]
sh scripts/train_video.sh [CONFIG] [NUM_GPUS] [BATCH_SIZE] [OUTPUT_DIR] [OPTS]

# Training on LVIS dataset with ResNet50 backbone
sh scripts/train.sh configs/clipvis_R50.yaml 4 8 output/lvis MODEL.MASK_FORMER.DEC_LAYERS 7
#Training on COCO dataset with ResNet50 backbone
sh scripts/train.sh configs/clipvis_R50.yaml 4 8 output/coco MODEL.MASK_FORMER.DEC_LAYERS 10 DATASETS.TRAIN '("coco_2017_train",)' DATASETS.TEST '("coco_2017_val",)'
#Fine-tune on YTVIS2019 dataset with ResNet50 backbone
sh scripts/train_video.sh configs/clipvis_video_R50.yaml 4 8 output/ytvis MODEL.MASK_FORMER.DEC_LAYERS 10 MODEL.WEIGHTS models/coco/model_final.pth
```

## Evaluation
We provide shell scripts ```scripts/eval_video.sh``` for Evaluation on various video datasets. 
 

### Evaluation script
```bash
sh scripts/eval_video.sh [CONFIG] [NUM_GPUS] [VAL_DATA] [TEST_NUM_CLASS] [OUTPUT_DIR] [WEIGHTS] [OPTS]

#Evaluation on validation set of LV-VIS datset
sh scripts/eval_video.sh configs/clipvis_video_R50.yaml 4 '("lvvis_val",)' 1196 output/lvvis models/clipvis_lvis_r50_7.pth MODEL.MASK_FORMER.DEC_LAYERS 7
```

## Results

We train our network on training set of LVIS dataset and evaluate our network on multiple video datasets. We provide pretrained weights for our models reported in the paper. All of the models were evaluated with 4 NVIDIA 3090 GPUs.


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Training Data</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">LV-VIS val</th>
<th valign="bottom">LV-VIS test</th>
<th valign="bottom">OVIS</th>
<th valign="bottom">YTVIS19</th>
<th valign="bottom">YTVIS21</th>
<th valign="bottom">BURST</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: CLIPVIS (R) -->
<tr>
<td align="center">LVIS</td>
<td align="center">ResNet-50</td>
<td align="center">18.8</td>
<td align="center">13.8</td>
<td align="center">12.3</td>
<td align="center">30.7</td>
<td align="center">28.7</td>
<td align="center">5.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1NhjFlRO9UPUtgzyFKCnxjCDlVqiPhu7Q/view?usp=sharing">ckpt</a>&nbsp;
</tr>
<!-- ROW: CLIPVIS (B) -->
<tr>

<td align="center">LVIS</td>
<td align="center">ConvNeXt-B</td>
<td align="center">32.1</td>
<td align="center">25.2</td>
<td align="center">18.2</td>
<td align="center">42.3</td>
<td align="center">39.5</td>
<td align="center">8.5</td>
<td align="center"><a href="https://drive.google.com/file/d/1kq83mvHqKZP6nYvo3hbBNu9sIPWos5JN/view?usp=sharing">ckpt</a>&nbsp;
</tr>

</tbody></table>

## Citation

```BibTeX
@misc{zhu2024clipvis,
      title={CLIP-VIS: Adapting CLIP for Open-Vocabulary Video Instance Segmentation}, 
      author={Wenqi Zhu, Jiale Cao, Jin Xie, Shuangming Yang, and Yanwei Pang},
      year={2024},
      eprint={2403.12455},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [Mask2Former](https://github.com/facebookresearch/Mask2Former), [LVVIS](https://github.com/haochenheheda/LVVIS/) and [fc-clip](https://github.com/bytedance/fc-clip) whose code has been utilized in this repository.
