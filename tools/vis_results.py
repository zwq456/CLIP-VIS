import os
import glob
import json
import cv2
import tqdm
from pycocotools import mask as pymask
import numpy as np
import tqdm
import argparse


def get_center(mask):
    # Get the central part of the object
    h1,h2 = np.argwhere(mask.sum(axis=1).reshape(-1)).min(), np.argwhere(mask.sum(axis=1).reshape(-1)).max()
    w1,w2 = np.argwhere(mask.sum(axis=0).reshape(-1)).min(), np.argwhere(mask.sum(axis=0).reshape(-1)).max()
    return int((h1+h2)/2), int((w1+w2)/2), h1, w1, h2, w2
def vis_re(dt_json,anno_json,img_dir,output_dir):
    color_map = [[20,255,20], [20, 20, 255], [255, 20, 20], [20, 255, 255], [255,20,255], [255,255,20],[42,42,128],[165, 42, 42], [134, 134, 103], [0, 0, 142], [255, 109, 65], \
        [0, 226, 252], [5, 121, 0], [0, 60, 100], [250, 170, 30], [100, 170, 30], [179, 0, 194], [255, 77, 255], [120, 166, 157], \
        [73, 77, 174], [0, 80, 100], [182, 182, 255], [0, 143, 149], [174, 57, 255], [0, 0, 230], [72, 0, 118], [255, 179, 240], \
        [0, 125, 92], [209, 0, 151], [188, 208, 182], [145, 148, 174], [106, 0, 228], [0, 0, 70], [199, 100, 0], [166, 196, 102], \
        [110, 76, 0], [133, 129, 255], [0, 0, 192], [183, 130, 88], [130, 114, 135], [107, 142, 35], [0, 228, 0], [174, 255, 243], [255, 208, 186]]
    
    dt = json.load(open(dt_json, 'r'))
    data = json.load(open(anno_json, 'r'))
    categories = data['categories']

    videos = data['videos']
    # dt= data['annotations']

    dt_dic = {}
    category_dic = {}
    for category in categories:
        category_dic[category['id']] = category['name']

    for d in dt:
        if d['video_id'] not in dt_dic.keys():
            dt_dic[d['video_id']] = []
        dt_dic[d['video_id']].append(d)
    for video in tqdm.tqdm(videos):
        video_name=os.path.dirname(video['file_names'][0])
        img_list = video['file_names']
        img_list.sort()
        video_id = video['id']
        video_dt = dt_dic[video_id]
        for fid, img_path in enumerate(img_list):
            img = cv2.imread(os.path.join(img_dir, img_path))
            h,w,_ = img.shape
            mask_vis = np.zeros((h,w,3))
            for obj_id, obj in enumerate(video_dt):
                category_id = obj['category_id']
                category_name = category_dic[category_id]
                if 'score' in obj.keys():
                    score = obj['score']
                    if score < 0.2:
                        continue
                if obj['segmentations'][fid] is None:
                    continue
                obj_mask = pymask.decode(obj['segmentations'][fid])
                if obj_mask.sum() == 0:
                    continue
                color = color_map[int(obj_id)%len(color_map)]
                mask_vis[obj_mask > 0] = color
                img[obj_mask > 0] = img[obj_mask > 0] * 0.55 + mask_vis[obj_mask>0]*0.45
                contours,hierarchy = cv2.findContours(obj_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                img = cv2.drawContours(img,contours,-1,(222,222,222),2)
                h_,w_,y1,x1,y2,x2 = get_center(obj_mask)
                img = cv2.putText(img, category_name, ((x1 + x2)//2 , (y1+y2)//2 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                img = cv2.putText(img, category_name, ((x1 + x2)//2 , (y1+y2)//2 ), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

            img_name = img_path.split('/')[-1]
            os.makedirs(os.path.join(output_dir, video_name),exist_ok=True)
            output_path = os.path.join(output_dir, video_name, img_name)
            cv2.imwrite(output_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dn', default='ovis')
    parser.add_argument('--dj', default='output/debug/inference/results.json')
   
    args = parser.parse_args()
    if args.dn=='lvvis':
        output_dir = 'output/lvvis_vis'
        anno_json = 'datasets/lvvis/val/val_instances_.json'
        img_dir = 'datasets/lvvis/val/JPEGImages/'
    elif args.dn=='ovis':
        output_dir = 'output/ovis_vis_ad'
        anno_json = 'datasets/ovis/annotations/annotations_valid.json'
        img_dir = 'datasets/ovis/valid/'
    elif args.dn=='yt19':
        output_dir = 'output/yt19_vis'
        anno_json = 'datasets/ytvis2019/valid.json'
        img_dir = 'datasets/ytvis2019/valid/JPEGImages/'
    elif args.dn=='yt21':
        output_dir = 'output/yt21_vis'
        anno_json = 'datasets/ytvis2021/valid.json'
        img_dir = 'datasets/ytvis2021/valid/JPEGImages/'
    elif args.dn=='burst':    
        output_dir = 'output/burst_vis'
        anno_json = 'datasets/burst/b2y_val.json'
        img_dir = 'datasets/burst/val'

    dt_json = args.dj
    vis_re(dt_json,anno_json,img_dir,output_dir)




