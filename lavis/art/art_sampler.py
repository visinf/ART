from typing import Dict, List
import random
import copy
from collections import defaultdict
from lavis.art.utils import sample_items
import os, pickle
import numpy as np
import scipy.stats as stats
import torch

class ARTSampler:
    def __init__(self, cfg):  #, predicate_list, initial_pool, logger=None):
        self.cfg = cfg
        self.total_train_count = 0
        self.cum_al_budget = 0
        self.diversity_enabled_rel = []
        self.remove_from_pool = {}
        random.seed(self.cfg.seed)

    def balanced_sample(self, annotations: Dict,
                        num_predicates: int = 50) -> Dict[str, List[str]]:
        """
                Perform balanced (or random) sampling from training data.
                Returns a dict: {predicate_id: [sample_ids]}.
                """
        p_count_dict = dict.fromkeys(range(1, num_predicates + 1), 0)
        sampled_img_rel_idx_dict = dict.fromkeys(range(1, num_predicates + 1), None)

        if self.cfg.al.al_type != 'random':
            img_rel_idx_dict = dict.fromkeys(range(1, num_predicates + 1), None)
        else:
            img_rel_idx_dict = []

        self.total_train_count = 0

        # Count occurrences and prepare candidate lists
        for img_id, ann_dict in annotations.items():
            for i, flag in enumerate(ann_dict['text_output_flag']):
                if flag != 'Yes':
                    continue
                predicate = ann_dict['predicates'][i]
                p_count_dict[predicate] += 1
                self.total_train_count += 1

                if self.cfg.al.al_type != 'random':
                    rel_id = f"{img_id}_{i}"
                    if img_rel_idx_dict[predicate] is None:
                        img_rel_idx_dict[predicate] = [rel_id]
                    else:
                        img_rel_idx_dict[predicate].append(rel_id)
                else:
                    rel_id = f"{predicate}_{img_id}_{i}"
                    img_rel_idx_dict.append(rel_id)

        # Determine budget
        self.budget_per_loop = int(self.total_train_count * self.cfg.al.al_budget_per_loop)
        if self.cfg.al.al_type == 'random':
            random_samples = random.sample(img_rel_idx_dict, k=self.budget_per_loop)
            return self._group_by_predicate(random_samples)
        else:
            items, quantity = sample_items(p_count_dict, self.budget_per_loop)
            for predicate_id, count in quantity.items():
                sampled = random.sample(img_rel_idx_dict[predicate_id], k=count)
                sampled_img_rel_idx_dict[predicate_id] = sampled
        self.cum_al_budget += self.cfg.al.al_budget_per_loop
        return  sampled_img_rel_idx_dict

    def adaptive_sample_from_pool(self, pool_log, val_log, dataset, al_loop, output_folder):
        # === Step 1: Save Evaluation Output ===
        gt = pool_log[1]
        pred = pool_log[2]

        gt['rel'] = [t[2] for t in gt['relation_tuple_list']]
        data_dict = {'gt': gt, 'pred': pred}

        with open(os.path.join(output_folder, f'al_loop_{al_loop}.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)

        # === Step 2: Extract positive sample stats from pool ===
        pool_ann = dataset['pool'].annotation
        img_rel_idx_dict = dict.fromkeys(range(1, 51))
        p_count_dict = dict.fromkeys(range(1, 51), 0)
        self.total_train_count = 0

        for key in pool_ann:
            ann_dict = pool_ann[key]
            for i, flag in enumerate(ann_dict['text_output_flag']):
                if flag == 'Yes':
                    k = ann_dict['predicates'][i]
                    p_count_dict[k] += 1
                    self.total_train_count += 1
                    if img_rel_idx_dict[k] is None:
                        img_rel_idx_dict[k] = [f'{key}_{i}']
                    else:
                        img_rel_idx_dict[k].append(f'{key}_{i}')

        # === Step 3: Compute predicate priority scores from validation recalls ===
        predictions = val_log["stats"][-1][1]  # Recall per predicate
        total_score = sum(predictions) if sum(predictions) != 0 else 1e-8
        priority_scores = {
            i + 1: (1 - p) / total_score if p != 0 else 1 / total_score for i, p in enumerate(predictions)
        }

        # === Step 4: Prepare uncertainty metrics ===
        total_existing_samples = dict.fromkeys(range(1, 51))
        uncertainty_dict = {}
        samples_for_diversity = dict.fromkeys(range(1, 51))

        for rel in img_rel_idx_dict.keys():
            pos = np.where((np.array(pred[0]) == np.array(gt['text_output'])) & (np.array(gt['rel']) == rel))[
                0].tolist()
            neg = [i for i, p_ in enumerate(pred[0]) if p_.startswith('No') and (gt['rel'][i] == rel)]
            unsure_pos = [i for i in range(len(pred[0])) if i not in pos and i not in neg and (gt['rel'][i] == rel)]
            total_existing_samples[rel] = len(pos) + len(neg) + len(unsure_pos)
            samples_for_diversity[rel] = np.where((np.array(gt['rel']) == rel))[0].tolist()
            uncertainty_dict[rel] = {'pos': pos, 'neg': neg, 'unsure_pos': unsure_pos}

        classes_with_missing_samples = [class_index for class_index, total_samples in total_existing_samples.items() if
                                        total_samples == 0]

        for val in classes_with_missing_samples:
            priority_scores[val] = 0

        total_priority = sum(priority_scores.values())
        priority_scores_normalized = {index: score / total_priority for index, score in priority_scores.items()}


        self.cum_al_budget += self.cfg.al.al_budget_per_loop

        samples_per_class = dict.fromkeys(priority_scores, 0)
        for class_index, prediction in enumerate(predictions):
            samples_to_pick = priority_scores_normalized[class_index + 1] * self.budget_per_loop
            samples_per_class[class_index + 1] += int(samples_to_pick)

        additional_samples_needed = 0
        for class_index in sorted(samples_per_class.keys(), reverse=True):
            if total_existing_samples[class_index] is not None:
                existing_samples = total_existing_samples[class_index]
                samples_needed = samples_per_class[class_index] - existing_samples
                if samples_needed > 0:
                    samples_per_class[class_index] = existing_samples
                    additional_samples_needed += samples_needed

        total_samples_per_class = [x for x in total_existing_samples.values()]
        total_samples_ = sum(total_samples_per_class)
        priority_based_on_sample_dist = [x / total_samples_ for x in total_samples_per_class]

        for class_index in samples_per_class.keys():
            samples_per_class[class_index] += int(
                additional_samples_needed * priority_based_on_sample_dist[class_index - 1])

        leftover = self.budget_per_loop - sum([x for x in samples_per_class.values()])
        if leftover > 0:
            max_class = priority_based_on_sample_dist.index(max(priority_based_on_sample_dist)) + 1
            samples_per_class[max_class] += leftover

        current_samples_selected = 0
        pos_entropy_count = 0
        neg_entropy_head_count = 0
        neg_entropy_tail_count = 0
        pred_idx_unsure_count = 0
        selected_pred_indices = []
        self.remove_from_pool={}
        # === Step 5: Perform uncertainty dissimilarity sampling per predicate ===
        for rel, rel_budget in samples_per_class.items():
            if rel_budget > 0:
                if rel not in self.diversity_enabled_rel:
                    selection = self._sample_uncertain_dissimilar_instances(rel, rel_budget, pred, uncertainty_dict)
                    selected_indices = selection["all"]
                    #self.remove_from_pool[rel] = selected_indices

                    current_samples_selected_ = len(selected_indices)
                    pos_entropy_count_ = len(selection['pos'])
                    neg_entropy_head_count_ = len(selection['neg_head'])
                    neg_entropy_tail_count_ = len(selection['neg_tail'])
                    pred_idx_unsure_count_ = len(selection['unsure'])

                    current_samples_selected += current_samples_selected_
                    pos_entropy_count += pos_entropy_count_
                    neg_entropy_head_count += neg_entropy_head_count_
                    neg_entropy_tail_count += neg_entropy_tail_count_
                    pred_idx_unsure_count += pred_idx_unsure_count_
                else:
                    selected_indices = random.sample(samples_for_diversity[rel], rel_budget)
                    #self.remove_from_pool[rel] = selected_indices

                selected_pred_indices.extend(selected_indices)
        # Match samples and mark them for transfer from pool to train
        self._match_and_mark_samples_for_transfer(
            dataset=dataset,
            pred_idx=selected_pred_indices,  # from uncertainty sampler
            s_pred_img_ids=pred[2],
            s_pred_s_box=pred[3],
            s_pred_o_box=pred[4],
            s_gt_output=gt['text_output'],
            al_loop=al_loop,
            output_folder=output_folder
        )
        # Finalize the transfer of marked samples and update the dataset
        self._finalize_sample_transfer(dataset)
        self._clean_empty_pool_entries(dataset)
        return samples_per_class, dataset, current_samples_selected, \
       pos_entropy_count, neg_entropy_head_count, \
       neg_entropy_tail_count, pred_idx_unsure_count

    def _clean_empty_pool_entries(self, dataset):
        """
        Removes annotation entries from the pool whose text_output is empty,
        and syncs the 'key' list accordingly.
        """
        pool_ann = dataset['pool'].annotation
        pool_keys = dataset['pool'].key

        keys_to_remove = [k for k in list(pool_ann.keys()) if pool_ann[k]['text_output'] == []]
        for k in keys_to_remove:
            del pool_ann[k]
            if k in pool_keys:
                pool_keys.remove(k)

    def _sample_uncertain_dissimilar_instances(self, rel, rel_budget, pred, uncertainty_dict):
        # Extract uncertainty dissimilarity bins: TP as pos, FN as neg, FP as unsure
        pos = uncertainty_dict[rel]['pos']
        neg = uncertainty_dict[rel]['neg']
        unsure = uncertainty_dict[rel]['unsure_pos']

        pos_entropy = np.array(pred[5])[pos].tolist()
        neg_entropy = np.array(pred[5])[neg].tolist()
        unsure_seps = np.array(pred[6])[unsure].tolist()

        if not pos and not neg and not unsure:
            return {"all": [], "pos": [], "neg_head": [], "neg_tail": [], "unsure": []}

        # Estimate normal distributions for each uncertainty metric for adaptive thresholding
        def norm_dist(vals):
            if len(vals) < 2:
                return stats.norm(0, 1)
            return stats.norm(np.mean(vals), np.std(vals) + 1e-6)

        pos_d = norm_dist(pos_entropy)
        neg_d = norm_dist(neg_entropy)
        unsure_d = norm_dist(unsure_seps)

        z = 1.96  # Initial z-score threshold for filtering samples with high/low uncertainty

        samples_pos_entropy = []
        samples_neg_entropy_head = []
        samples_neg_entropy_tail = []
        samples_unsure_seps = []

        while True:
            hp, tp = pos_d.mean() + z * pos_d.std(), pos_d.mean() - z * pos_d.std()
            hn, tn = neg_d.mean() + z * neg_d.std(), neg_d.mean() - z * neg_d.std()
            hu, tu = unsure_d.mean() + z * unsure_d.std(), unsure_d.mean() - z * unsure_d.std()

            samples_pos_entropy_ = [(i, val) for i, val in enumerate(pos_entropy) if val > hp]
            samples_neg_entropy_head_ = [(i, val) for i, val in enumerate(neg_entropy) if
                                         val > hn and (i, val) not in samples_neg_entropy_head]
            samples_neg_entropy_tail_ = [(i, val) for i, val in enumerate(neg_entropy) if
                                         val < tn and (i, val) not in samples_neg_entropy_head_ and (i,
                                                                                                     val) not in samples_neg_entropy_tail]
            samples_unsure_seps_ = [(i, val) for i, val in enumerate(unsure_seps) if
                                    val < tu and (i, val) not in samples_unsure_seps]

            if samples_pos_entropy_:
                s_pos_entropy_idx_, _ = zip(*samples_pos_entropy_)
            else:
                s_pos_entropy_idx_ = []
            if samples_neg_entropy_head_:
                s_neg_entropy_head_idx_, _ = zip(*samples_neg_entropy_head_)
            else:
                s_neg_entropy_head_idx_ = []
            if samples_neg_entropy_tail_:
                s_neg_entropy_tail_idx_, _ = zip(*samples_neg_entropy_tail_)
            else:
                s_neg_entropy_tail_idx_ = []
            if samples_unsure_seps_:
                s_unsure_seps_idx_, _ = zip(*samples_unsure_seps_)
            else:
                s_unsure_seps_idx_ = []

            total_samples = len(s_pos_entropy_idx_) + len(s_neg_entropy_head_idx_) + len(s_neg_entropy_tail_idx_) + len(
                s_unsure_seps_idx_)

            if total_samples > rel_budget:
                extra = total_samples - rel_budget
                remove_list = [0] * 4
                current_list = [len(s_pos_entropy_idx_), len(s_neg_entropy_head_idx_), len(s_neg_entropy_tail_idx_),
                                len(s_unsure_seps_idx_)]
                while extra > 0:
                    for l in range(4):
                        if current_list[l] > 0:
                            remove_list[l] += 1
                            current_list[l] -= 1
                            extra -= 1
                        if extra == 0:
                            break
                samples_pos_entropy_ = sorted(samples_pos_entropy_, key=lambda x: x[1], reverse=True)[
                                       :-remove_list[0]] if remove_list[0] else samples_pos_entropy_
                samples_neg_entropy_head_ = sorted(samples_neg_entropy_head_, key=lambda x: x[1], reverse=True)[
                                            :-remove_list[1]] if remove_list[1] else samples_neg_entropy_head_
                samples_neg_entropy_tail_ = sorted(samples_neg_entropy_tail_, key=lambda x: x[1])[:-remove_list[2]] if \
                remove_list[2] else samples_neg_entropy_tail_
                samples_unsure_seps_ = sorted(samples_unsure_seps_, key=lambda x: x[1])[:-remove_list[3]] if \
                remove_list[3] else samples_unsure_seps_

            elif total_samples < rel_budget and z < -1000:
                curr_budget = rel_budget - total_samples
                remaining_pos = [(i, val) for i, val in enumerate(pos_entropy) if (i, val) not in samples_pos_entropy_]
                remaining_neg = [(i, val) for i, val in enumerate(neg_entropy) if
                                 (i, val) not in samples_neg_entropy_head_ and (i,
                                                                                val) not in samples_neg_entropy_tail_]
                remaining_unsure = [(i, val) for i, val in enumerate(unsure_seps) if
                                    (i, val) not in samples_unsure_seps_]
                total_remaining = len(remaining_pos) + len(remaining_neg) + len(remaining_unsure)

                if total_remaining > 0:
                    ratio_pos = len(remaining_pos) / total_remaining
                    ratio_neg = len(remaining_neg) / total_remaining
                    ratio_unsure = len(remaining_unsure) / total_remaining

                    samples_pos_entropy += random.sample(remaining_pos,
                                                         min(len(remaining_pos), int(ratio_pos * curr_budget)))
                    samples_neg_entropy_head += random.sample(remaining_neg,
                                                              min(len(remaining_neg), int(ratio_neg * curr_budget)))
                    samples_unsure_seps += random.sample(remaining_unsure,
                                                         min(len(remaining_unsure), int(ratio_unsure * curr_budget)))

            if total_samples >= rel_budget or z < -1000:
                samples_pos_entropy.extend(samples_pos_entropy_)
                samples_neg_entropy_head.extend(samples_neg_entropy_head_)
                samples_neg_entropy_tail.extend(samples_neg_entropy_tail_)
                samples_unsure_seps.extend(samples_unsure_seps_)
                break
            else:
                z -= 0.1

        def extract(pred_indices, source):
            return [source[i] for i in pred_indices] if pred_indices else []

        s_pos_entropy_idx, _ = zip(*samples_pos_entropy) if samples_pos_entropy else ([], [])
        s_neg_entropy_head_idx, _ = zip(*samples_neg_entropy_head) if samples_neg_entropy_head else ([], [])
        s_neg_entropy_tail_idx, _ = zip(*samples_neg_entropy_tail) if samples_neg_entropy_tail else ([], [])
        s_unsure_seps_idx, _ = zip(*samples_unsure_seps) if samples_unsure_seps else ([], [])

        pred_idx_all = list(
            set(extract(s_pos_entropy_idx, pos) + extract(s_neg_entropy_head_idx, neg) + extract(s_neg_entropy_tail_idx,
                                                                                                 neg) + extract(
                s_unsure_seps_idx, unsure)))

        return {
            "all": pred_idx_all,
            "pos": extract(s_pos_entropy_idx, pos),
            "neg_head": extract(s_neg_entropy_head_idx, neg),
            "neg_tail": extract(s_neg_entropy_tail_idx, neg),
            "unsure": extract(s_unsure_seps_idx, unsure)
        }

    def _finalize_sample_transfer(self, dataset):
        for k in self.remove_from_pool:
            self.remove_from_pool[k] = sorted(self.remove_from_pool[k], reverse=True)

        for k in self.remove_from_pool.keys():
            for idx in self.remove_from_pool[k]:
                for key in dataset['pool'].annotation[k].keys():
                    if key not in ['filename', 'img_info', 'labels', 'labels_num', 'bbox', 'distance']:
                        if type(dataset['train']).__name__ == 'ConcatDataset':
                            dataset['train'].datasets[0].annotation[k][key].append(
                                dataset['pool'].annotation[k][key][idx])
                            dataset['pool'].annotation[k][key].pop(idx)
                        else:
                            dataset['train'].annotation[k][key].append(dataset['pool'].annotation[k][key][idx])
                            dataset['pool'].annotation[k][key].pop(idx)

            if dataset['pool'].annotation[k]['text_output'] == []:
                del dataset['pool'].annotation[k]
                dataset['pool'].key.pop(dataset['pool'].key.index(k))

    def _match_and_mark_samples_for_transfer(self, dataset, pred_idx, s_pred_img_ids, s_pred_s_box, s_pred_o_box,
                                             s_gt_output, al_loop, output_folder):
        """
        Matches selected predictions with their annotations and records indices for removal from pool.

        Args:
            dataset (dict): Dataset dict with 'pool' and 'train'.
            pred_idx (list[int]): Indices of selected predictions.
            s_pred_img_ids (list): Corresponding image IDs.
            s_pred_s_box (list): Subject boxes.
            s_pred_o_box (list): Object boxes.
            s_gt_output (list): Ground truth sentences.
            al_loop (int): AL loop number for debugging.
            output_folder (str): Path to dump debug info.
        """
        pool_ann = dataset['pool'].annotation
        keys = list(pool_ann.keys())
        im_id_keys = [pool_ann[k]['img_info']['image_id'] for k in keys]
        key_list_train = dataset['train'].datasets[0].annotation.keys() if isinstance(dataset['train'],
                                                                                      torch.utils.data.ConcatDataset) else \
        dataset['train'].annotation.keys()
        train_ann = dataset['train'].datasets[0].annotation if isinstance(dataset['train'],
                                                                          torch.utils.data.ConcatDataset) else dataset[
            'train'].annotation
        s_pred_img_ids = np.array(s_pred_img_ids)[pred_idx].tolist()
        zipped_s_pred_img_ids = list(zip(s_pred_img_ids, pred_idx))
        sorted_s_pred_img_ids, sorted_pred_idx = zip(*sorted(zipped_s_pred_img_ids, key=lambda x: x[0]))
        sorted_pred_idx = list(sorted_pred_idx)
        s_pred_s_box = np.array(s_pred_s_box)[sorted_pred_idx].tolist()
        s_pred_o_box = np.array(s_pred_o_box)[sorted_pred_idx].tolist()
        s_gt_output = np.array(s_gt_output)[sorted_pred_idx].tolist()

        id_counter = 0
        for k, im_d in zip(keys, im_id_keys):
            if id_counter >= len(sorted_s_pred_img_ids):
                break
            while id_counter < len(sorted_s_pred_img_ids) and sorted_s_pred_img_ids[id_counter] == im_d:
                text_output = pool_ann[k]['text_output']
                so_box_pair_array = np.array(pool_ann[k]['so_box_pair']).reshape(len(pool_ann[k]['so_box_pair']), -1)

                indices = np.where(
                    np.all(so_box_pair_array == np.array(s_pred_s_box[id_counter] + s_pred_o_box[id_counter]), axis=1)
                )[0].tolist()

                if len(so_box_pair_array) != len(text_output):
                    print('Length mismatch')
                    print(f'annotation:{k}')

                for idx in indices:
                    if idx >= len(text_output):
                        print(s_pred_s_box[id_counter] + s_pred_o_box[id_counter])
                        print(f'annotation:{k}')
                        with open(os.path.join(output_folder, f'al_dataset_train_debug_{al_loop}.pth'), 'wb') as f:
                            torch.save(self.datasets, f)
                        break

                    if text_output[idx] == s_gt_output[id_counter]:
                        key_list = list(pool_ann[k].keys())

                        if k not in key_list_train:
                            im_id_exists_in_train = False
                            train_ann[k] = {kk: None for kk in key_list}
                            for key in key_list:
                                if key in ['filename', 'img_info', 'labels', 'labels_num', 'bbox', 'distance']:
                                    train_ann[k][key] = pool_ann[k][key]
                                else:
                                    if not im_id_exists_in_train:
                                        indices_neg = np.where(np.array(self.orig_ann[k]['text_output_flag']) == 'No')[
                                            0].tolist()
                                        train_ann[k][key] = np.array(self.orig_ann[k][key])[indices_neg].tolist()

                        if k in self.remove_from_pool:
                            self.remove_from_pool[k].append(idx)
                        else:
                            self.remove_from_pool[k] = [idx]

                        id_counter += 1

    def _build_train_idx_dict(self, sampled_dict):
        """
            sampled_dict: either
              - {predicate → [img_idx_rel_idx]} for even/balanced
              - [pred_img_idx_rel_idx] for random

            Returns: {img_idx → [rel_idx]}
            """
        train_idx_dict = defaultdict(list)

        if self.cfg.al.al_type == "random":
            for val in sampled_dict:
                pred, img_idx, rel = map(int, val.split("_"))
                train_idx_dict[img_idx].append(rel)
        else:
            for pred, entries in sampled_dict.items():
                if entries is not None:
                    for val in entries:
                        img_idx, rel = map(int, val.split("_"))
                        train_idx_dict[img_idx].append(rel)

        return train_idx_dict

    def _move_samples_to_train(self, dataset):
        """Move selected samples from pool to train and clean up pool annotations."""

        # Sort the idx in descending order for each key to pop correctly from pool
        for k in self.remove_from_pool:
            self.remove_from_pool[k] = sorted(self.remove_from_pool[k], reverse=True)

        for k in self.remove_from_pool.keys():
            for idx in self.remove_from_pool[k]:
                for key in dataset['pool'].annotation[k].keys():
                    if key not in ['filename', 'img_info', 'labels', 'labels_num', 'bbox', 'distance']:
                        if type(dataset['train']).__name__ == 'ConcatDataset':
                            dataset['train'].datasets[0].annotation[k][key].append(
                                dataset['pool'].annotation[k][key][idx]
                            )
                        else:
                            dataset['train'].annotation[k][key].append(
                                dataset['pool'].annotation[k][key][idx]
                            )
                        dataset['pool'].annotation[k][key].pop(idx)

        # Remove empty annotations from pool
        empty_keys = []
        for k in dataset['pool'].annotation:
            if dataset['pool'].annotation[k]['text_output'] == []:
                empty_keys.append(k)
        for k in empty_keys:
            del dataset['pool'].annotation[k]
            dataset['pool'].key.remove(k)

    def _allocate_budget(self, priority_scores, available_counts, total_budget):
        sorted_preds = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        allocated = dict.fromkeys(priority_scores.keys(), 0)
        index = 0
        while total_budget > 0:
            pred_id = sorted_preds[index][0]
            if available_counts[pred_id] > 0:
                allocated[pred_id] += 1
                available_counts[pred_id] -= 1
                total_budget -= 1
            index = (index + 1) % len(sorted_preds)
        return allocated

