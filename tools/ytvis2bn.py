import json
import os
from detectron2.utils.file_io import PathManager
import argparse

novel_cate=[ "earless", "seal" ,"fox" ,"leopard" ,"snake", "ape", "hand", "sedan","flying" ,"dsic" ,"whale"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vt', default='datasets/ytvis2019/valid.json')
    parser.add_argument('--rt', default='output/yt19-b8-24w-out/inference/ytvis_2019_val/results.json')

    args = parser.parse_args()
    val_data=json.load(open(args.vt))
    src_json=json.load(open(args.rt))
    class_name = {x['id']:x['name']  for x in val_data['categories']}
    
    base_results=[]
    novel_results=[]

    for ann in src_json:
        if class_name[ann["category_id"]] in novel_cate:
            novel_results.append(ann)
            # print(class_name[ann["category_id"]])
        else:
            base_results.append(ann)
    file_path_b = os.path.join(os.path.dirname(args.rt), "base_results.json")
    with PathManager.open(file_path_b, "w") as f:
        f.write(json.dumps(base_results))
        f.flush()

    file_path_n = os.path.join(os.path.dirname(args.rt), "novel_results.json")
    with PathManager.open(file_path_n, "w") as f:
        f.write(json.dumps(novel_results))
        f.flush()



    print("finished")