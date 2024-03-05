## Installation

### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.12 is recommended and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- `pip install -r requirements.txt`

An example of installation is shown below:

```
conda create --name clipvis python=3.8 -y
conda activate clipvis
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -U opencv-python

# under your working directory
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..

# install LVIS API
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/lvis-dataset/lvis-api.git

# clone this repo
git clone https://github.com/zwq456/CLIP-VIS.git.
cd CLIP-VIS
pip install -r requirements.txt
cd clipvis/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```
