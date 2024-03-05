from evaluate.lvvis import LVVIS
from evaluate.burst import BURST
from evaluate.bursteval import BURSTeval
from evaluate.lvviseval import LVVISeval

import sys
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import itertools
import torch 
from detectron2.utils.file_io import PathManager
import argparse
import json
from datetime import datetime

import os
import sys
import logging


def pth_to_json(pth_path):
    predictions=torch.load(pth_path)
    coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
    for result in coco_results:
        category_id = result["category_id"]
        result["category_id"] = category_id+1
    file_path = os.path.join(os.path.dirname(pth_path), "instances_results.json")
    with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps(coco_results))
                    f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', default='output/burst-res-openai/vote/9t7/inference/results.json')
    parser.add_argument('--et', default='("lvvis_val",)')

    args = parser.parse_args()
    dt_path=os.path.join(args.dt,'inference/results.json')
    output_file = os.path.join(os.path.dirname(args.dt), "results.txt")
    logging.basicConfig(filename=output_file, level=logging.INFO, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')


    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)
    eval_type=args.et

    if 'lvvis' in eval_type:
        DATAEVAL=LVVIS
        DATAEVALeval=LVVISeval
        gt_path='datasets/lvvis/val/val_instances_.json'

    elif 'burst' in eval_type:
        DATAEVAL=BURST
        DATAEVALeval=BURSTeval
        gt_path='datasets/burst/b2y_val.json'
    else:
        logger.info("\n")
        logger.info(f"\nAnnotations is invalid\n")
        raise NotImplementedError
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("\n")
    logger.info(f"\n===== {current_time} =====\n")
    ytvosGT = DATAEVAL(gt_path)
    ytvosDT = ytvosGT.loadRes(dt_path)
    ytvosEval = DATAEVALeval(ytvosGT, ytvosDT, "segm")
    ytvosEval.evaluate()
    ytvosEval.accumulate()
    ytvosEval.summarize()
  