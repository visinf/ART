import os
import numpy as np
import torch
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.eval_core import run_evaluation
import torch.distributed as dist
from lavis.common.dist_utils import main_process


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, dataset_name=None, report_metric=True):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.dataset_name = dataset_name
        self.report_metric = report_metric
        self.predictions = None
        self.dataset = None
        self.gr_enabled = False
    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(
            num_beams=run_cfg.num_beams,
            max_len=run_cfg.max_len,
            min_len=run_cfg.min_len,
            evaluate=run_cfg.evaluate,
            report_metric=run_cfg.get("report_metric", True),
            dataset_name=run_cfg.get("dataset_name", None)
        )

    def valid_step(self, model, samples, split_name=None):
        results = []
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            split_name=split_name,
        )
        if self.dataset_name in ['gqa', 'oi']:
            self.gr_enabled = True
        self.gr_flag = []
        if split_name == 'val' and self.dataset_name in ['gqa', 'oi']:
            for i, (caption, gt, sim_score) in enumerate(zip(captions[0], samples["text_output"], captions[2])):
                if not caption.startswith("No "):
                    if sim_score >= 0.95 and gt.lower() != caption.lower():
                        self.gr_flag.append(1)
                        captions[0][i] = gt
                    else:
                        self.gr_flag.append(0)
        else:
            self.gr_flag = [0] * len(captions[0])

        caption_list = list(captions[0])
        if split_name == 'pool':
            result_rel = list(range(len(caption_list)))
        else:
            result_rel = [i for i, cap in enumerate(caption_list) if not cap.startswith("No ")]

        conf = captions[1].cpu()[result_rel]
        entropy = captions[2].cpu()[result_rel] if split_name == 'pool' else None

        cap = [caption_list[i] for i in result_rel]

        if self.predictions is None:
            self.predictions = [cap, conf,
                                np.array(samples['image_id_list'])[result_rel].tolist(),
                                np.array(samples['s_box_list'])[result_rel].tolist(),
                                np.array(samples['o_box_list'])[result_rel].tolist()]
            if split_name == 'pool':
                self.predictions.extend([entropy.tolist(), captions[3]])
            elif split_name == 'val':
                self.predictions.extend([captions[2], self.gr_flag, np.array(samples['relation_tuple_list'])[result_rel].tolist()])
        else:
            self.predictions[0].extend(cap)
            self.predictions[1] = torch.cat((self.predictions[1], conf), dim=0)
            for idx, key in enumerate(range(2, 5)):
                self.predictions[key].extend(np.array(samples[['image_id_list', 's_box_list', 'o_box_list'][idx]])[result_rel].tolist())
            if split_name == 'pool':
                self.predictions[5].extend(entropy.tolist())
                self.predictions[6].extend(captions[3])
            elif split_name == 'val':
                self.predictions[5].extend(captions[2])
                self.predictions[6].extend(self.gr_flag)
                self.predictions[7].extend(np.array(samples['relation_tuple_list'])[result_rel].tolist())

        if self.dataset is None:
            self.dataset = {}
            keys = ['text_output', 'image_id_list', 's_box_list', 'o_box_list', 'relation_tuple_list', 'rel_bool']
            for key in keys:
                val = samples[key]
                self.dataset[key] = np.array(val)[result_rel].tolist() if isinstance(val, list) else val
            keys = ['img_info', 'bbox', 'labels']
            for key in keys:
                self.dataset[key] = samples[key]
            self.dataset['image'] = self.dataset['prompt'] = self.dataset['image_region_list'] = None
        else:
            for key in ['text_output', 'image_id_list', 's_box_list', 'o_box_list', 'relation_tuple_list', 'rel_bool']:
                self.dataset[key].extend(np.array(samples[key])[result_rel].tolist())
            for key in ['img_info', 'bbox', 'labels']:
                self.dataset[key].extend(samples[key])

        if split_name == 'pool':
            return 0, self.dataset, self.predictions

        for caption, img_id, gt_triplet in zip(captions[0], samples["image_id_list"], samples["text_output"]):
            if not caption.startswith("No "):
                results.append({"caption": caption, "image_id": int(img_id), "gt": gt_triplet})

        return results, self.dataset, self.predictions



    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        if val_result[1] is not None and val_result[2] is not None:
            dataset, predictions = val_result[1], val_result[2]
            output_folder = registry.get_path("result_dir")
            result_file = os.path.join(output_folder, f"test_result_{self.dataset_name}.pth")

            torch.save(val_result, result_file)

            if len(predictions) > 0:
                avg_metrics, result_dict_list_to_log = run_evaluation(
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    split_name=split_name,
                    epoch=epoch, gr_enabled=self.gr_enabled,dataset_type=self.dataset_name
                )
            else:
                avg_metrics, result_dict_list_to_log = 0.0, None

            val_log = {
                "agg_metrics": avg_metrics,
                "stats": result_dict_list_to_log
            }

            # Reset internal state
            self.dataset = None
            self.predictions = None

            return val_log

        return {"agg_metrics": 0.0}


