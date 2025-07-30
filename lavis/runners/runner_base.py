"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import datetime
import json
import logging
import os
import random
import time
from pathlib import Path
import pickle
import fcntl
import copy

import torch
import torch.distributed as dist
import webdataset as wds
from lavis.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
    is_dist_avail_and_initialized,
)
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from lavis.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset
from rtpt import RTPT
from lavis.art.art_sampler import ARTSampler
from lavis.art.utils_distributed import sync_scalar_across_ranks
from lavis.art.utils import split_train_pool_annotations

@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0

        # self.setup_seeds()
        self.setup_output_dir()
        random.seed(self.config.run_cfg.seed)
        self.art_sampler = ARTSampler(self.config.run_cfg)

    def log_active_learning_state(self, output_folder, cur_epoch, al_loop, budget_per_loop, cum_al_budget):
        al_cfg = self.config.run_cfg.al
        info = (
            f"{str(al_cfg)}\n"
            f"al_loop: {al_loop}\n"
            f"budget_per_loop: {budget_per_loop}\n"
            f"cum_al_budget: {cum_al_budget}\n"
            f"train_stop: {self.train_stop}\n"
        )
        with open(os.path.join(output_folder, "evaluation_res.txt"), 'a') as f:
            f.write(info + "\n")

    def gather_and_merge_distributed_objects(self, local_obj, split_name="pool"):
        """
        Gathers and merges distributed objects across all ranks.

        Args:
            local_obj: The local object to gather (e.g., pool_log or results)
            mode (str): "pool" or "results" — determines how [0] is merged

        Returns:
            merged object aggregated from all ranks
        """
        assert split_name in {"pool", "val"}, f"Invalid mode: {split_name}"

        world_size = dist.get_world_size()
        object_list = [None] * world_size
        dist.all_gather_object(object_list, local_obj)

        for i, obj in enumerate(object_list):
            if i == 0:
                merged = copy.deepcopy(obj)
            else:
                if split_name == "val":
                    merged[0].extend(obj[0])  # Merge [0] only in results mode

                for key in obj[1]:
                    if obj[1][key] is not None:
                        merged[1][key].extend(obj[1][key])

                for j in range(len(merged[2])):
                    if j != 1:
                        merged[2][j].extend(obj[2][j])
                    else:
                        merged[2][j] = torch.cat((merged[2][j], obj[2][j]))

        return merged

    def should_run_adaptive_sampling(self, train_stop, cum_al_budget):
        return (
                train_stop == 3
                and cum_al_budget < self.config.run_cfg.al.total_al_budget
                and self.config.run_cfg.al.al_enabled
        )

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu], find_unused_parameters=True
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            lr_scale = self.config.run_cfg.get("lr_layer_decay", 1)
            weight_decay = self.config.run_cfg.get("weight_decay", 0.05)
            optim_params = self._model.get_optimizer_params(weight_decay, lr_scale)

            num_parameters = 0
            for p_group in optim_params:
                for p in p_group["params"]:
                    num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: {}".format(num_parameters))

            beta2 = self.config.run_cfg.get("beta2", 0.999)

            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                betas=(0.9, beta2),
            )
        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            # max_epoch = self.config.run_cfg.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.run_cfg.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run_cfg.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self, al_info=None) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """

        if self._dataloaders is None:
            print(f"Process {dist.get_rank()} line 216.")
            #pid = os.fork()
            # reoganize datasets by split and concatenate/chain if necessary
            dataset_ratios = self.config.run_cfg.get("train_dataset_ratios", None)

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            datasets = reorg_datasets_by_split(self.datasets)
            self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                        self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                dataset_ratios=dataset_ratios,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]
        return train_dataloader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        #if not self.evaluate_only and self.resume_ckpt_path is not None:
        #    self._load_checkpoint(self.resume_ckpt_path)
        rtpt = RTPT(name_initials='GS', experiment_name='f5_random_balanced_gqa', max_iterations=self.max_epoch)
        rtpt.start()
        ###variables
        self.train_stop = 0
        self.diversity_enabled_rel = []
        prev_val = 0
        curr_val = 100
        cum_al_budget = 0
        al_loop = 0
        best_agg_metric = 0
        random.seed(self.config.run_cfg.seed)
        #is_best = True
        combined_results = []
        # dict.fromkeys(range(self.start_epoch, self.max_epoch), [])
        if not self.evaluate_only:
            for cur_epoch in range(self.start_epoch, self.max_epoch):
                # training phase
                logging.info("Start training")
                if self.config.run_cfg.al.al_enabled:
                    if cur_epoch == 0:
                        al_loop = 1
                        logging.info(f"[AL] Running balanced sampling for epoch {cur_epoch}")
                        self.al_loop = 1  # start AL loop counter

                        # 1. Deepcopy annotation from the training dataset
                        data_train = (self.datasets['vg_instruct_sgg']['train']
                                      if 'vg_instruct_sgg' in self.datasets
                                      else self.datasets['train'].datasets[0])
                        annotations = copy.deepcopy(data_train.annotation)
                        #orig_ann = annotations  # keep original for slicing later
                        self.art_sampler.orig_ann = self.orig_ann = copy.deepcopy(data_train.annotation)
                        # 2. Run balanced sampling
                        sampled_img_rel_idx_dict = self.art_sampler.balanced_sample(
                            annotations=annotations)

                        # 3. Convert sampled entries to train_idx_dict
                        train_idx_dict = self.art_sampler._build_train_idx_dict(
                            sampled_img_rel_idx_dict
                        )

                        # 4. Split annotation into AL train / pool annotations
                        al_train_dict, pool_dict = split_train_pool_annotations(self.orig_ann, train_idx_dict)

                        # 5. Update LAVIS datasets
                        self.update_lavis_datasets(al_train_dict, pool_dict)

                        self.art_sampler._clean_empty_pool_entries(self.datasets['vg_instruct_sgg'])

                    if self.use_distributed:
                        dist.barrier()
                    train_stats = self.train_epoch(cur_epoch, al_info=self.config.run_cfg.al)
                    output_folder = registry.get_path("result_dir")
                    budget_per_loop = self.art_sampler.budget_per_loop
                    cum_al_budget = self.art_sampler.cum_al_budget
                    if is_main_process():
                        if is_main_process() and cur_epoch == 0:
                            self.log_active_learning_state(
                                output_folder=output_folder,
                                cur_epoch=cur_epoch,
                                al_loop=al_loop,
                                budget_per_loop=budget_per_loop,
                                cum_al_budget=cum_al_budget
                            )

                else:
                    train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)
                rtpt.step()


                # evaluation phase
                if len(self.valid_splits) > 0 and cur_epoch % 1 == 0:
                    for split_name in self.valid_splits:
                        logging.info("Evaluating on {}.".format(split_name))
                        #if not self.use_distributed:
                        #    val_log = self.eval_epoch(
                        #        split_name=split_name, cur_epoch=cur_epoch
                        #    )
                        #else:
                        results = self.eval_epoch(
                            split_name=split_name, cur_epoch=cur_epoch
                        )
                        if self.use_distributed:
                            results = self.gather_and_merge_distributed_objects(results, split_name=split_name)

                        if is_main_process():
                            #if self.use_distributed:
                            val_log = self.task.after_evaluation(results, split_name, cur_epoch)
                            assert (
                                    "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."
                            agg_metrics = val_log["agg_metrics"]
                            prev_val = best_agg_metric
                            if agg_metrics > best_agg_metric and split_name == "val":
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics
                                #self._save_checkpoint(cur_epoch, is_best=True)

                            val_log.update({"best_epoch": best_epoch})
                            self.log_stats(val_log, split_name)
                            if self.config.run_cfg.al.al_enabled:

                                curr_val = agg_metrics
                                progress_diff = (curr_val - prev_val)
                                self.train_stop = self.train_stop + 1 if progress_diff < 0.01 else 0

                            if self.should_run_adaptive_sampling(self.train_stop, cum_al_budget):
                                al_loop += 1
                                logging.info(f"[AL] Starting adaptive loop {al_loop}")

                        if self.use_distributed:
                            self.train_stop = sync_scalar_across_ranks(self.train_stop)
                            al_loop = sync_scalar_across_ranks(al_loop)
                            dist.barrier()

                        if self.train_stop == 3:
                            pool_log = self.eval_epoch(split_name="pool", cur_epoch=cur_epoch)
                            if self.use_distributed:
                                pool_log = self.gather_and_merge_distributed_objects(pool_log, split_name="pool")
                        if is_main_process() and self.train_stop == 3:
                            if self.config.run_cfg.al.al_type == 'art':
                                # Run adaptive sampling to select new samples and move them from pool to train
                                samples_per_class, self.datasets, current_samples_selected, \
                                    pos_entropy_count, neg_entropy_head_count, \
                                    neg_entropy_tail_count, pred_idx_unsure_count = self.art_sampler.adaptive_sample_from_pool(
                                    pool_log=pool_log,
                                    val_log=val_log,
                                    dataset=self.datasets,
                                    al_loop=al_loop,
                                    output_folder=registry.get_path("result_dir")
                                )

                            # === Backing up datasets ===
                            self._backup_and_switch_datasets(output_folder)
                            cum_al_budget += self.config.run_cfg.al.al_budget_per_loop

                            # reinitialize the model for from scratch training
                            prev_val = 0
                            curr_val = 100
                            best_agg_metric = 0

                    if self.train_stop == 3:
                        pretrained_model = "checkpoint to the pretrained model"
                        self._reinitialize_model_from_checkpoint(output_folder, pretrained_model)

                    # === Logging ===
                    if is_main_process():

                        al_info = 'al_loop: ' + str(
                            al_loop) + '\n' + 'budget_per_loop: ' + str(
                            budget_per_loop) + '\n' + 'cum_al_budget: ' + str(
                            cum_al_budget) + '\n' + 'train_stop: ' + str(self.train_stop) + '\n' + 'prev_val: ' + str(
                            prev_val) + '\n' + 'curr_val: ' + str(curr_val) + '\n' + 'progress_diff: ' + str(
                            progress_diff)
                        if al_loop > 1 and self.config.run_cfg.al.al_type == 'art':
                            al_info = al_info + '\n' + 'current_samples_selected: ' + str(
                                current_samples_selected) + ', pos_entropy_count: ' + str(
                                pos_entropy_count) + ', neg_entropy_head_count: ' + str(
                                neg_entropy_head_count) + ', neg_entropy_tail_count: ' + str(
                                neg_entropy_tail_count) + ', pred_idx_unsure_count: ' + str(pred_idx_unsure_count)



                        with open(os.path.join(output_folder, "evaluation_res.txt"), 'a') as f:
                            f.write(al_info + '\n')
                    dist.barrier()
                    if cum_al_budget >= self.config.run_cfg.al.total_al_budget and self.train_stop == 3:  # stop training
                        break

                    if self.use_distributed:
                        dist.barrier()
        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        split_name = "val"
        logging.info("Evaluating on {}.".format(split_name))
        results = self.eval_epoch(
            split_name=split_name, cur_epoch=test_epoch, skip_reload=self.evaluate_only
        )
        if self.use_distributed:
            results = self.gather_and_merge_distributed_objects(results, split_name=split_name)
            dist.barrier()
        if is_main_process():
            self.task.after_evaluation(results, split_name, test_epoch)
        if self.use_distributed:
            dist.barrier()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def update_lavis_datasets(self, al_train_dict, pool_dict):
        if 'vg_instruct_sgg' in self.datasets:
            self.datasets['vg_instruct_sgg']['train'].annotation = al_train_dict
            self.datasets['vg_instruct_sgg']['train'].key = list(al_train_dict.keys())
            self.datasets['vg_instruct_sgg']['pool'].annotation = pool_dict
            self.datasets['vg_instruct_sgg']['pool'].key = list(pool_dict.keys())
        else:
            self.datasets['train'].datasets[0].annotation = al_train_dict
            self.datasets['train'].datasets[0].key = list(al_train_dict.keys())
            self.datasets['pool'].annotation = pool_dict
            self.datasets['pool'].key = list(pool_dict.keys())

    def train_epoch(self, epoch, al_info=None):
        # train
        self.model.train()
        # self.epoch = epoch
        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()

        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results, gt, pred = self.task.evaluation(model, data_loader, split_name=split_name)
        results = [results, gt, pred]
        if results[0] is not None and not self.use_distributed:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )

        else:
            return results

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
            self,
            datasets,
            num_workers,
            batch_sizes,
            is_trains,
            collate_fns,
            dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                    dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
                datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        #for k in list(state_dict.keys()):
        #    if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                #del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    def _backup_and_switch_datasets(self, output_folder):
        if isinstance(self.datasets['train'], object):
            if type(self.datasets['train']).__name__ == 'ConcatDataset':
                backup = {
                    'train': self.datasets['train'].datasets[0],
                    'val': self.datasets['val'],
                    'test': self.datasets['test'],
                    'pool': self.datasets['pool'],
                    'train_backup': self.datasets['train_backup']
                }
        else:
            backup = {
                'train': self.datasets['train'],
                'val': self.datasets['val'],
                'test': self.datasets['test'],
                'pool': self.datasets['pool'],
                'train_backup': self.datasets['train_backup']
            }

        self.datasets = {'vg_instruct_sgg': backup}
        if len(self.datasets) == 1:
            new_train = self.datasets['vg_instruct_sgg']['train']
            self.datasets['vg_instruct_sgg']['train'] = self.datasets['vg_instruct_sgg']['train_backup']
            self.datasets['vg_instruct_sgg']['train'].annotation = new_train.annotation
            self.datasets['vg_instruct_sgg']['train'].key = list(
                self.datasets['vg_instruct_sgg']['train'].annotation.keys())
        else:
            self.datasets['train'].datasets[0].key = list(self.datasets['train'].datasets[0].annotation.keys())

        with open(os.path.join(output_folder, f'al_dataset_train.pth'), 'wb') as f:
            torch.save(self.datasets, f)

    def _reinitialize_model_from_checkpoint(self, output_folder, checkpoint_path):
        self._dataloaders = None
        if self.use_distributed and not is_main_process():
            self.datasets = None
        dist.barrier()
        if dist.get_rank() != 0:
            with open(os.path.join(output_folder, f'al_dataset_train.pth'), 'rb') as f:
                self.datasets = torch.load(f)

        print(f"Reinitialize the model.....Process {dist.get_rank()} loading pretrained.")
        self._load_checkpoint(checkpoint_path)
        self.train_stop = 0

    @main_process
    def log_stats(self, stats, split_name=None):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:

                f.write(json.dumps(log_stats) + "\n")

        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
