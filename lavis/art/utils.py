
def sample_items(p_count_dict, budget):
    """
    Distributes `budget` samples evenly over items in `p_count_dict`.
    Returns a list of selected items and a dict of quantities per item.
    """
    item_freq_dict = p_count_dict.copy()
    sorted_items = sorted(item_freq_dict.items(), key=lambda x: x[1], reverse=True)
    selected_items = []
    selected_quantity_dict = {}
    index = 0
    while budget > 0:
        item = sorted_items[index % len(sorted_items)][0]
        frequency = item_freq_dict[item]
        if frequency > 0:
            selected_items.append(item)
            selected_quantity_dict[item] = selected_quantity_dict.get(item, 0) + 1
            item_freq_dict[item] -= 1
            budget -= 1
        index += 1
    return selected_items, selected_quantity_dict

def split_train_pool_annotations(orig_ann, train_idx_dict):
    """
    Splits annotations into train and pool based on selected relation indices.

    - Train = selected (positive) or automatically included false examples
    - Pool = leftover true examples
    Only includes images in `al_train_idx_dict` if they were selected for training
    """
    al_train_idx_dict = {}
    pool_idx_dict = {}

    for img_id, ann in orig_ann.items():
        so_len = len(ann['so_pair'])

        if img_id not in train_idx_dict:
            # Still need to add to pool if true rels exist
            pool_idxs = [i for i in range(so_len) if ann['rel_bool_at_type_level'][i]]
            if pool_idxs:
                pool_idx_dict[img_id] = _slice_ann(ann, pool_idxs)
            continue  # ❌ Skip adding to train dict

        selected_idxs = train_idx_dict[img_id]

        # TRAIN = selected rels + false rels
        al_idxs = [i for i in range(so_len) if i in selected_idxs or not ann['rel_bool_at_type_level'][i]]
        if al_idxs:
            al_train_idx_dict[img_id] = _slice_ann(ann, al_idxs)

        # POOL = true rels that weren't selected
        pool_idxs = [i for i in range(so_len) if i not in selected_idxs and ann['rel_bool_at_type_level'][i]]
        if pool_idxs:
            pool_idx_dict[img_id] = _slice_ann(ann, pool_idxs)

    return al_train_idx_dict, pool_idx_dict


def _slice_ann(ann, indices):
    return {
        k: v if k in ['filename', 'img_info', 'labels', 'labels_num', 'bbox', 'distance']
        else [v[i] for i in indices]
        for k, v in ann.items()
    }