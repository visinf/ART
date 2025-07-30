# eval_core.py
import os
import torch
from collections import Counter
from tqdm import tqdm
import json
import numpy as np
from lavis.tasks.sgg_eval import SGRecall, SGZeroShotRecall, SGMeanRecall
# Optionally import others as needed

SUPPORTED_DATASETS = {"vg", "gqa", "oi"}

def run_evaluation(
    dataset=None,
    predictions=None,
    output_folder=None,
    split_name=None,
    epoch=None,
    mode="predcls",
    iou_types=["relations"],
    gr_enabled=True,
    attribute_on=False,
    num_attributes=0,
    multiple_preds=False,
    iou_thres=0.5,
    dataset_type="vg"
):
    assert dataset_type in SUPPORTED_DATASETS, f"Unsupported dataset type: {dataset_type}"

    # === PREPARE VOCAB ===
    dataset['ind_to_classes'], dataset['ind_to_predicate'] = _get_vocab(dataset_type)

    # Parse model outputs into prediction tensors
    (pred_triplets, pred_rel_scores, pred_rel_labels,
     pred_rel_inds, pred_triplet_boxes, pred_image_ids, pred_gr_flags) = _parse_predictions(dataset, predictions)

    # Prepare per-image prediction and ground-truth dictionaries
    predictions, predictions_gr, groundtruths = _build_image_level_data(
        dataset=dataset,
        pred_triplets=pred_triplets,
        pred_rel_scores=pred_rel_scores,
        pred_rel_labels=pred_rel_labels,
        pred_rel_inds=pred_rel_inds,
        pred_triplet_boxes=pred_triplet_boxes,
        pred_image_ids=pred_image_ids,
        pred_gr_flags=pred_gr_flags,
        gr_enabled=gr_enabled
    )

    # Run evaluation and return metrics
    avg_metrics, result_logs = evaluate_predictions(
        dataset=dataset,
        predictions=predictions,
        predictions_gr=predictions_gr,
        groundtruths=groundtruths,
        output_folder=output_folder,
        split_name=split_name,
        epoch=epoch,
        mode=mode,
        iou_types=iou_types,
        num_rel_category=len(dataset['ind_to_predicate']),
        ind_to_predicates=dataset['ind_to_predicate'],
        attribute_on=attribute_on,
        num_attributes=num_attributes,
        multiple_preds=multiple_preds,
        iou_thres=iou_thres,
        gr_enabled=gr_enabled
    )

    return avg_metrics, result_logs

def load_cate_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if '__background__' not in info['rel']:
        ind_to_predicates_cate = ['__background__'] + info['rel']
    else:
        ind_to_predicates_cate = info['rel']
    if '__background__' not in info['obj']:
        ind_to_entites_cate = ['__background__'] + info['obj']
    else:
        ind_to_entites_cate = info['obj']

    # print(len(ind_to_predicates_cate))
    # print(len(ind_to_entites_cate))
    predicate_to_ind = {idx: name for idx, name in enumerate(ind_to_predicates_cate)}
    entites_cate_to_ind = {idx: name for idx, name in enumerate(ind_to_entites_cate)}

    return (ind_to_entites_cate, ind_to_predicates_cate,
            entites_cate_to_ind, predicate_to_ind)



def _get_vocab(dataset_type):
    if dataset_type == "vg":
        vg_dict = "data/vg/VG_100K/stanford_filtered/stanford_vg/VG-SGG-dicts-with-attri.json"
        with open(vg_dict, 'r') as f:
            data_dict = json.load(f)
        ind_to_predicates = ['__background__'] + [data_dict['idx_to_predicate'][k] for k in data_dict['idx_to_predicate']]
        ind_to_classes = ['__background__'] + [data_dict['idx_to_label'][k] for k in data_dict['idx_to_label']]
    elif dataset_type == "oi":
        dict_file = "data/open-imagev4/annotations/categories_dict.json"
        (ind_to_classes, ind_to_predicates,
         classes_to_ind, predicates_to_ind) = load_cate_info(dict_file)
        ind_to_predicates = [p.lower() for p in ind_to_predicates]
        ind_to_classes = [c.lower() for c in ind_to_classes]
    elif dataset_type == "gqa":
        dict_file = 'data/GQA/GQA_200_ID_Info.json'
        info = json.load(open(dict_file, 'r'))
        ind_to_predicates = info['ind_to_predicates']  # info['ind_to_predicates']  #['__background__']
        ind_to_classes = info['ind_to_classes']
    else:
        raise NotImplementedError(dataset_type)
    return ind_to_classes, ind_to_predicates


def generate_eval_res_dict(evaluator, mode):
    res_dict = {}
    for k, v in evaluator.result_dict[f'{mode}_{evaluator.type}'].items():
        res_dict[f'{mode}_{evaluator.type}/top{k}'] = np.mean(v)
    return res_dict

def evaluate_predictions(
    dataset,
    predictions,
    predictions_gr,
    groundtruths,
    output_folder,
    split_name,
    epoch,
    mode,
    iou_types,
    num_rel_category,
    ind_to_predicates,
    attribute_on=False,
    num_attributes=0,
    multiple_preds=False,
    iou_thres=0.5,
    gr_enabled=True
):
    result_str = '\n' + '=' * 100 + '\n'
    result_str += f"Epoch: {epoch}, Split: {split_name} \n"
    result_dict_list_to_log = []
    avg_metrics = None

    if "relations" in iou_types:
        from lavis.tasks.sgg_eval import SGRecall, SGMeanRecall

        rel_eval_result_dict = {}
        evaluator = {}

        eval_recall = SGRecall(rel_eval_result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        eval_mean_recall = SGMeanRecall(rel_eval_result_dict, num_rel_category, ind_to_predicates, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        if gr_enabled:
            eval_gr_recall = SGRecall(rel_eval_result_dict)
            eval_gr_recall.register_container(mode)
            evaluator['eval_gr_recall'] = eval_gr_recall

            eval_gr_mean_recall = SGMeanRecall(rel_eval_result_dict, num_rel_category, ind_to_predicates, print_detail=True)
            eval_gr_mean_recall.register_container(mode)
            evaluator['eval_gr_mean_recall'] = eval_gr_mean_recall

        global_container = {
            'result_dict': rel_eval_result_dict,
            'mode': mode,
            'multiple_preds': multiple_preds,
            'num_rel_category': num_rel_category,
            'iou_thres': iou_thres,
            'attribute_on': attribute_on,
            'num_attributes': num_attributes
        }


        for gt, pred, pred_gr in tqdm(zip(groundtruths, predictions, predictions_gr if gr_enabled else predictions), total=len(predictions)):
            evaluate_relation_of_one_image(gt, pred, global_container, evaluator, prediction_gr=pred_gr if gr_enabled else None)

        eval_mean_recall.calculate_mean_recall(mode)
        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)

        if gr_enabled:
            eval_gr_mean_recall.calculate_mean_recall(mode, gr=True)
            result_str += eval_gr_recall.generate_print_string(mode, gr=True)
            result_str += eval_gr_mean_recall.generate_print_string(mode, gr=True)

        class_level_recall = eval_mean_recall.get_class_level_recall(mode)
        result_dict_list_to_log.extend([
            generate_eval_res_dict(eval_recall, mode),
            generate_eval_res_dict(eval_mean_recall, mode),
            class_level_recall
        ])

        if avg_metrics is None:
            avg_metrics = (np.mean(rel_eval_result_dict[mode + '_recall'][100]) + rel_eval_result_dict[mode + '_mean_recall'][100]) / 3

        if output_folder:
            rel_eval_result_dict['ind_to_classes'] = dataset['ind_to_classes']
            rel_eval_result_dict['ind_to_predicate'] = dataset['ind_to_predicate']
            torch.save(rel_eval_result_dict, os.path.join(output_folder, 'result_dict.pytorch'))

    if output_folder:
        with open(os.path.join(output_folder, "evaluation_res.txt"), 'a') as f:
            f.write(result_str + '\n')

    return float(avg_metrics), result_dict_list_to_log


def _parse_predictions(dataset, predictions_):
    stripped_prediction = [p.replace(' <s>', '') if ' <s>' in p else p for p in predictions_[0]]
    stripped_prediction_ = [p.replace('Yes, ', '') if 'Yes, ' in p else p for p in stripped_prediction]

    s_str = [p.split(" ")[0] for p in stripped_prediction_]
    p_str = [' '.join(p.split(" ")[1:-1]) for p in stripped_prediction_]
    o_str = [p.split(" ")[-1] for p in stripped_prediction_]

    s = [dataset['ind_to_classes'].index(p_) if p_ in dataset['ind_to_classes'] else 0 for p_ in s_str]
    p = [dataset['ind_to_predicate'].index(p_) if p_ in dataset['ind_to_predicate'] else 0 for p_ in p_str]
    o = [dataset['ind_to_classes'].index(p_) if p_ in dataset['ind_to_classes'] else 0 for p_ in o_str]

    pred_triplets = torch.tensor([[s_, o_, p_] for s_, o_, p_ in zip(s, o, p)], dtype=torch.float32)
    pred_rel_scores = torch.tensor(predictions_[1])
    pred_rel_labels = torch.tensor(p, dtype=torch.float32)
    pred_rel_inds = torch.tensor([[s_, o_] for s_, o_ in zip(s, o)])
    pred_triplet_boxes = torch.column_stack((torch.tensor(predictions_[3]), torch.tensor(predictions_[4])))
    pred_image_ids = torch.tensor(predictions_[2], dtype=torch.int32)
    pred_gr_flags = torch.tensor(predictions_[6], dtype=torch.bool) if len(predictions_) > 6 else torch.zeros_like(pred_image_ids, dtype=torch.bool)

    return (pred_triplets, pred_rel_scores, pred_rel_labels, pred_rel_inds,
            pred_triplet_boxes, pred_image_ids, pred_gr_flags)


def _build_image_level_data(dataset, pred_triplets, pred_rel_scores, pred_rel_labels,
                             pred_rel_inds, pred_triplet_boxes, pred_image_ids, pred_gr_flags,
                             gr_enabled=True):
    image_id_tensor = torch.tensor(dataset['image_id_list'], dtype=torch.int32)
    rel_bool = torch.tensor(dataset['rel_bool'], dtype=torch.bool)
    gt_triplets = torch.tensor(dataset['relation_tuple_list'], dtype=torch.int64)
    gt_triplet_boxes = torch.column_stack((torch.tensor(dataset["s_box_list"]), torch.tensor(dataset["o_box_list"])))

    predictions, predictions_gr, groundtruths = [], [], []
    img_ids = set(dataset['image_id_list'])
    img_splits = [dataset['img_info'][i]['image_id'] for i in range(len(dataset['img_info']))]
    img_splits_counter = Counter(img_splits)
    start = 0

    for image_id, orig_img_id in enumerate(img_ids):
        predictions.append({})
        groundtruths.append({})
        if gr_enabled:
            predictions_gr.append({})

        mask = pred_image_ids == orig_img_id
        if gr_enabled:
            mask_nongr = mask & (~pred_gr_flags)
        else:
            mask_nongr = mask

        predictions[-1]['bbox'] = None
        predictions[-1]['img_info'] = dataset['img_info'][start]
        predictions[-1]['pred_triplets'] = pred_triplets[mask_nongr]
        predictions[-1]['pred_rel_scores'] = pred_rel_scores[mask_nongr]
        predictions[-1]['pred_rel_labels'] = pred_rel_labels[mask_nongr]
        predictions[-1]['pred_rel_inds'] = pred_rel_inds[mask_nongr]
        predictions[-1]['pred_triplet_boxes'] = pred_triplet_boxes[mask_nongr]

        if gr_enabled:
            predictions_gr[-1]['bbox'] = None
            predictions_gr[-1]['img_info'] = dataset['img_info'][start]
            predictions_gr[-1]['pred_triplets'] = pred_triplets[mask]
            predictions_gr[-1]['pred_rel_scores'] = pred_rel_scores[mask]
            predictions_gr[-1]['pred_rel_labels'] = pred_rel_labels[mask]
            predictions_gr[-1]['pred_rel_inds'] = pred_rel_inds[mask]
            predictions_gr[-1]['pred_triplet_boxes'] = pred_triplet_boxes[mask]

        image_mask = image_id_tensor == orig_img_id
        groundtruths[-1]['img_info'] = dataset['img_info'][start]
        groundtruths[-1]['bbox'] = torch.tensor(dataset['bbox'][start], dtype=torch.float32)
        groundtruths[-1]['labels'] = torch.tensor(dataset['labels'][start], dtype=torch.int64)
        groundtruths[-1]['rel_bool'] = rel_bool[image_mask]
        groundtruths[-1]['gt_triplet_boxes'] = gt_triplet_boxes[image_mask & rel_bool]
        groundtruths[-1]['gt_triplets'] = gt_triplets[image_mask & rel_bool]

        start += img_splits_counter[orig_img_id]

    return predictions, predictions_gr if gr_enabled else None, groundtruths

def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, prediction_gr=None):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    #local_container['gt_rels'] = groundtruth['relation_tuple'].long().detach().cpu().numpy()
    local_container['gt_triplets'] = groundtruth['gt_triplets'].detach().cpu().numpy()
    local_container['gt_triplet_boxes'] = groundtruth['gt_triplet_boxes'].detach().cpu().numpy()
    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_triplets']) == 0:
        return

    local_container['gt_boxes'] = groundtruth['bbox'].detach().cpu().numpy()  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth['labels'].long().detach().cpu().numpy()  # (#gt_objs, )


    local_container['pred_rel_inds'] = prediction['pred_rel_inds'].long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction['pred_rel_scores'].detach().cpu().numpy()  # (#pred_rels, num_pred_class)



    local_container['pred_triplets'] = prediction['pred_triplets'].detach().cpu().numpy()
    local_container['pred_triplet_boxes'] = prediction['pred_triplet_boxes'].detach().cpu().numpy()
    #evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    #evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    if prediction_gr is not None:
        local_container['pred_rel_inds'] = prediction_gr['pred_rel_inds'].long().detach().cpu().numpy()  # (#pred_rels, 2)
        local_container['rel_scores'] = prediction_gr['pred_rel_scores'].detach().cpu().numpy()  # (#pred_rels, num_pred_class)

        local_container['pred_triplets'] = prediction_gr['pred_triplets'].detach().cpu().numpy()
        local_container['pred_triplet_boxes'] = prediction_gr['pred_triplet_boxes'].detach().cpu().numpy()
        local_container = evaluator['eval_gr_recall'].calculate_recall(global_container, local_container, mode, gr=True)
        evaluator['eval_gr_mean_recall'].collect_mean_recall_items(global_container, local_container, mode, gr=True)
    return
