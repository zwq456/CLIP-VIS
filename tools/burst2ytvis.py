import json
import argparse
import os
import json
import pycocotools.mask as cocomask
from tabulate import tabulate
from typing import Union
import copy

def _global_track_id(*, local_track_id: Union[str, int],
                     video_id: Union[str, int],
                     track_id_mapping) -> int:
    # remap local track ids into globally unique ids
    return track_id_mapping[str(video_id)][str(local_track_id)]


class B2YConverter:
    def __init__(self, b_format,class_split):
        self._b_format = b_format
        self._class_common_split=class_split
        self._split = b_format['split']
        self._ori_categories = b_format['categories']
        self._categories =self._make_categories()
        self._cate_map=self._make_map()
        self._videos = []
        self._annotations = []
        self._tracks = {}
        self._images = []
        self._next_img_id = 0
        self._next_ann_id = 0
        
        self._track_id_mapping = self._load_track_id_mapping()

        for seq in b_format['sequences']:
            self._visit_seq(seq)

    def _load_track_id_mapping(self):
        id_map = {}
        next_global_track_id = 1
        for seq in self._b_format['sequences']:
            seq_id = seq['id']
            seq_id_map = {}
            id_map[str(seq_id)] = seq_id_map
            for local_track_id in seq['track_category_ids']:
                seq_id_map[str(local_track_id)] = next_global_track_id
                next_global_track_id += 1
        return id_map

    def global_track_id(self, *, local_track_id: Union[str, int],
                        video_id: Union[str, int]) -> int:
        return _global_track_id(local_track_id=local_track_id,
                                video_id=video_id,
                                track_id_mapping=self._track_id_mapping)

    def _visit_seq(self, seq):
        self._make_video(seq)
        imgs = self._make_images(seq)
        self._make_annotations_and_tracks(seq, imgs)

    def _make_images(self, seq):
        imgs = []
        for img_path in seq['annotated_image_paths']:
            video = self._split + '/' + seq['dataset'] + '/' + seq['seq_name']
            file_name = video + '/' + img_path

            # TODO: once python 3.9 is more common, we can use this nicer and safer code
            #stripped = img_path.removesuffix('.jpg').removesuffix('.png').removeprefix('frame')
            stripped = img_path.replace('.jpg', '').replace('.png', '').replace('frame', '')

            last = stripped.split('_')[-1]
            frame_idx = int(last)

            img = {'id': self._next_img_id, 'video': video,
                   'width': seq['width'], 'height': seq['height'],
                   'file_name': file_name,
                   'frame_index': frame_idx,
                   'video_id': seq['id']}
            self._next_img_id += 1
            self._images.append(img)
            imgs.append(img)
        return imgs

    def _make_video(self, seq):
        video_id = seq['id']
        dataset = seq['dataset']
        seq_name = seq['seq_name']
        name =   dataset + '/' + seq_name
        file_name=[name+'/'+iname for iname in seq['annotated_image_paths']]
        video = {
            'id': video_id, 'width': seq['width'], 'height': seq['height'],'length':len(file_name),
            'neg_category_ids': seq['neg_category_ids'],
            'not_exhaustive_category_ids': seq['not_exhaustive_category_ids'],
            'file_names': file_name, 'metadata': {'dataset': dataset}}
        self._videos.append(video)

    def _make_annotations_and_tracks(self, seq, imgs):
        video_id = seq['id']
        segs = seq['segmentations']
        assert len(segs) == len(imgs), (len(segs), len(imgs))
        for i in seq['track_category_ids'].keys():
            segmentations=[]
            bboxs=[]
            for frame_segs, img in zip(segs, imgs):
                if i in frame_segs:
                    rle = frame_segs[i]['rle']
                    segment = {'counts': rle, 'size': [img['height'], img['width']]}
                    segmentations.append(segment)
                    coco_bbox = cocomask.toBbox(segment)
                    bbox = [int(x) for x in coco_bbox]
                    bboxs.append(bbox)
                else :
                    segmentations.append(None)
                    bboxs.append(None)
            category_id = int(seq['track_category_ids'][i])
            ann = {'segmentations': segmentations, 'id': self._next_ann_id,
                 'category_id': self._cate_map[category_id],'width': seq['width'], 'height': seq['height'],
                 'video_id': video_id,
                'bboxes': bboxs}
            self._next_ann_id += 1
            self._annotations.append(ann)

    def convert(self):
        return {'videos': self._videos, 'annotations': self._annotations,
                 'images': self._images,
                'categories': self._categories,
                'cate_ori':self._ori_categories,
                'track_id_mapping': self._track_id_mapping,
                'split': self._split}

    def _make_categories(self):
        common_class=self._class_common_split['common']
        uncommon_class=self._class_common_split['uncommon']
        cate_mod=[]
        for idx,cate in enumerate(self._ori_categories):
            cate_2=copy.deepcopy(cate)
            if cate_2['id'] in common_class:
                cate_2['split']='common'
            if cate_2['id'] in uncommon_class:
                cate_2['split']='uncommon'
            cate_2['id']=idx+1
            cate_mod.append(cate_2)
        
        return cate_mod
    def _make_map(self):

        cate_map={}
        for idx,(ori,mod) in enumerate(zip(self._ori_categories,self._categories)):
            cate_map[ori['id']]=mod['id']
        return cate_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', type=str,default='datasets/burst/val/all_classes.json')
    parser.add_argument('--out', type=str,default='datasets/burst/val/b2y_val.json')
    args = parser.parse_args()
    class_common='datasets/burst/info/class_split.json'
    with open(class_common) as ft:
        class_common_dict = json.load(ft)
    with open(args.ann) as f:
        b_format_gt = json.load(f)
    y_format_gt = B2YConverter(b_format_gt,class_common_dict).convert()
    with open(args.out, 'w') as f:
        json.dump(y_format_gt, f)
