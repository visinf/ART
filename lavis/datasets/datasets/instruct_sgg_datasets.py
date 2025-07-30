import os

from PIL import Image
import numpy as np
from lavis.datasets.datasets.base_dataset import BaseDataset
from random import sample
from lavis.datasets.datasets.vqa_datasets import VQADataset
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
import torch
import random

class InstructSGGDataset_samples_balanced(BaseDataset):
    def __init__(self, vis_processor, text_processor, ann_paths=None, vis_root=None):
        super().__init__()
        self.vis_processor = vis_processor
        self.file = ann_paths[0]
        with open(self.file, 'rb') as handle:
            self.annotation = pickle.load(handle)
        self.key = list(self.annotation.keys())
        self.text_processor = text_processor
        #self._add_instance_ids()
        self.bs = 100
        self.negative_ratio = 0.2
        random.seed(42)

    def evenly_sample_lists(self, list_p, list_n):

        # Calculate the number of samples to take from each list
        sample_ratio_list1 = 1 - self.negative_ratio #len(list_p) / total_length
        sample_ratio_list2 = self.negative_ratio #len(list_n) / total_length

        # Calculate the number of samples for each list based on the ratios
        if len(list_p) > 0:
            num_samples_list1 = min(int(sample_ratio_list1 * self.bs), len(list_p))
            num_samples_list2 = min(int(sample_ratio_list2 * self.bs), len(list_p))

        else:
            num_samples_list1 = 0
            if len(list_n) > 10:
                num_samples_list2 = 10
            else:
                num_samples_list2 = len(list_n)
        # Randomly sample from each list
        sampled_list1 = random.sample(list_p, num_samples_list1)
        if len(list_n) >= num_samples_list2:
            sampled_list2 = random.sample(list_n, num_samples_list2)

            # Combine the sampled lists
            combined_samples = sampled_list1 + sampled_list2
        else:
            combined_samples = sampled_list1

        return combined_samples, num_samples_list1, num_samples_list2

    def __getitem__(self, index):
        key = self.key[index]
        filename = self.annotation[key]['filename']
        image = Image.open(filename).convert("RGB")
        out = self.annotation[key]['text_output']
        flag = self.annotation[key]['text_output_flag']
        flag_arr = np.array(flag)

        no_rel = []
        rel = np.where(flag_arr != 'No')[0].tolist()

        no_rel_check =len(np.where(flag_arr == 'No')[0].tolist())
        if no_rel_check > 0:
            for val in rel:
                no_rel_curr = np.where((np.array(self.annotation[key]['p_associated_n']) ==
                                        self.annotation[key]['p_associated_n'][val]) & (flag_arr == 'No'))[0].tolist()
                no_rel.extend(random.sample(no_rel_curr, k=1))


        total = rel + no_rel
        p = len(rel)
        n = len(no_rel)
        image_region_list = None
        text_input = np.array(self.annotation[key]['text_input'])[total].tolist()
        text_output = np.array(self.annotation[key]['text_output'])[total].tolist()
        image_region = np.array(self.annotation[key]['union_box'])[total].tolist()

        for union_box in image_region:
            if image_region_list is None:
                image_region_list = [image.crop((union_box))]
            else:
                image_region_list.append(image.crop((union_box)))
        if image_region_list is None:
            print("Empty image region list")
        try:
            processed_img_reg_list = [self.vis_processor(img) for img in image_region_list]
        except:
            print(f"image_region_list: {image_region_list}")
            print(f"total: {total}")
            print(f"p: {p}")
            print(f"n: {n}")
        return {
            "image": processed_img_reg_list,
            "text_input": text_input,
            "text_output": text_output
        }


    def collater(self, samples):
        image_list, question_list, answer_list, weight_list, rel_list = [], [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.extend(sample["image"])
            question_list.extend(sample["text_input"])
            answer_list.extend(sample["text_output"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann['key'] = str(idx)


class InstructSGGEvalDataset_samples(BaseDataset):
    def __init__(self, vis_processor, text_processor, ann_paths=None, vis_root=None):
        super().__init__()
        self.vis_processor = vis_processor
        self.file = ann_paths[0]
        with open(self.file, 'rb') as handle:
            self.annotation = pickle.load(handle)
        self.key = list(self.annotation.keys())
        self.text_processor = text_processor
        self.bs = 35

    def __getitem__(self, index):
        key = self.key[index]
        filename = self.annotation[key]['filename']
        img_info = self.annotation[key]['img_info']
        bbox = self.annotation[key]['bbox']
        labels = self.annotation[key]['labels_num']

        # Load and transform the images
        image = Image.open(filename).convert("RGB")
        image_region_list = None
        if len(self.annotation[key]['text_input']) > self.bs:
            print(f"before reduction {len(self.annotation[key]['text_input'])}")
        text_input = self.annotation[key]['text_input'][:self.bs]
        text_output = self.annotation[key]['text_output'][:self.bs]
        image_region = self.annotation[key]['union_box'][:self.bs]
        image_id = self.annotation[key]['img_id'][:self.bs]
        relation_tuple = self.annotation[key]['gt_triplets'][:self.bs]
        s_boxes = self.annotation[key]['s_boxes'][:self.bs]
        o_boxes = self.annotation[key]['o_boxes'][:self.bs]
        rel_bool = self.annotation[key]['rel_bool_at_type_level'][:self.bs]

        for union_box in image_region:
            if image_region_list is None:
                image_region_list = [image.crop((union_box))]
            else:
                image_region_list.append(image.crop((union_box)))

        if image_region_list is None:
            print("I Reached image_region_list is None")
        processed_img_reg_list = [self.vis_processor(img) for img in image_region_list]


        return {
            "image": processed_img_reg_list,
            "text_input": text_input,
            "text_output": text_output,
            "image_id": image_id,
            "image_region": image_region,
            "img_info": img_info,
            "bbox": bbox,
            "labels": labels,
            "filename": filename,
            "relation_tuple": relation_tuple,
            "sbox": s_boxes,
            "obox": o_boxes,
            "rel_bool": rel_bool

        }

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list, image_region_list, image_id_list, img_info, bbox, filenames, relation_tuple_list, labels, s_box_list, o_box_list, rel_bool_list, flag = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


        for sample in samples:
            image_list.extend(sample["image"])
            question_list.extend(sample["text_input"])
            image_region_list.extend(sample["image_region"])
            image_id_list.extend(sample["image_id"])
            relation_tuple_list.extend(sample["relation_tuple"])
            s_box_list.extend(sample["sbox"])
            o_box_list.extend(sample["obox"])
            rel_bool_list.extend(sample["rel_bool"])
            answer_list.extend(sample["text_output"])
            img_info.append(sample["img_info"])
            bbox.append(sample["bbox"])
            filenames.append(sample["filename"])
            labels.append(sample["labels"])

        return {
            "image": torch.stack(image_list, dim=0),
            "prompt": question_list,
            "text_output": answer_list,
            "image_region_list": image_region_list,
            "image_id_list": image_id_list,
            "img_info": img_info,
            "bbox": bbox,
            "filename": filenames,
            "relation_tuple_list": relation_tuple_list,
            "labels": labels,
            "s_box_list": s_box_list,
            "o_box_list": o_box_list,
            "rel_bool": rel_bool_list
        }

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann['key'] = str(idx)
