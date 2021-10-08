import copy
import pathlib
import pickle
import random
import logging
import argparse
from typing import Tuple, Dict
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
import pandas as pd
import torch
from mergedeep import Strategy, merge
from typeguard import typechecked

import model as module_arch
import model.model_noise as module_noise
import model.metric as module_metric
import utils.visualizer as module_vis
import data_loader.data_loaders as module_data
from trainer import verbose, ctxt_mgr
from utils.util import compute_dims, compute_trn_config, update_src_web_video_dir
from parse_config import ConfigParser


@typechecked
def compress_predictions(query_masks: np.ndarray, sims: np.ndarray, topk: int = 10):
    """We store the indices of the top-k predictions, rather than the full similarity
    matrix, to reduce storage requirements.

    NOTE: The similarity matrix contains `num_queries x num_videos` elements, where
    `num_queries = num_videos x max_num_queries_per_video`.  We first mask out
    locations in the similarity matrix that correspond to invalid queries (these are
    produced by videos with fewer than `max_num_queries_per_video` descriptions).
    """

    # validate the input shapes
    assert query_masks.ndim == 2, "Expected query_masks to be a matrix"
    query_num_videos, query_max_per_video = query_masks.shape
    sims_queries, sims_num_videos = sims.shape
    msg = (f"Expected sims and query masks to represent the same number of videos "
           f"(found {sims_num_videos} v {query_num_videos}")
    assert query_num_videos == sims_num_videos, msg
    msg = (f"Expected sims and query masks to represent the same number of queries "
           f"(found {sims_queries} v {query_num_videos * query_max_per_video}")
    assert query_max_per_video * query_num_videos == sims_queries, msg

    valid_sims = sims[query_masks.flatten().astype(np.bool)]
    ranks = np.argsort(-valid_sims, axis=1)
    return ranks[:, :topk]


@typechecked
def get_model_and_data_loaders(
        config: ConfigParser,
        logger: logging.Logger,
        ckpt_path: Path,
        device: str
) -> Tuple[torch.nn.Module, module_data.ExpertDataLoader, torch.nn.Module]:
    expert_dims, raw_input_dims, text_dim = compute_dims(config)
    
    data_loaders = config.init(
        name='data_loader',
        module=module_data,
        logger=logger,
        raw_input_dims=raw_input_dims,
        challenge_mode=config.get("challenge_mode", False),
        text_dim=text_dim,
        text_feat=config["experts"]["text_feat"],
        text_agg=config["experts"]["text_agg"],
        use_zeros_for_missing=config["experts"].get("use_zeros_for_missing", False),
        task=config.get("task", "retrieval"),
        eval_only=True,
        distil_params=config.get("distil_params", None),
        training_file=config.get("training_file", None),
        testing_file=config.get("testing_file", None),
        caption_masks=config.get("caption_masks", None),
        ce_shared_dim=config["experts"].get("ce_shared_dim", None),
    )

    trn_config = compute_trn_config(config)
    model = config.init(
        name='arch',
        module=module_arch,
        trn_config=trn_config,
        expert_dims=expert_dims,
        text_dim=text_dim,
        disable_nan_checks=config["disable_nan_checks"],
        task=config.get("task", "retrieval"),
        ce_shared_dim=config["experts"].get("ce_shared_dim", None),
        feat_aggregation=config["data_loader"]["args"]["feat_aggregation"],
        trn_cat=config["data_loader"]["args"].get("trn_cat", 0),
    )
    model_noise = config.init(
        name='arch',
        module=module_noise,
        trn_config=trn_config,
        expert_dims=expert_dims,
        text_dim=text_dim,
        disable_nan_checks=config["disable_nan_checks"],
        task=config.get("task", "retrieval"),
        ce_shared_dim=config["experts"].get("ce_shared_dim", None),
        feat_aggregation=config["data_loader"]["args"]["feat_aggregation"],
        trn_cat=config["data_loader"]["args"].get("trn_cat", 0),
    )
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    ckpt_path = config._args.resume
    logger.info(f"Loading checkpoint: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    pathlib.PosixPath = temp
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    # support backwards compatibility
    deprecated = ["ce.moe_fc_bottleneck1", "ce.moe_cg", "ce.moe_fc_proj"]
    for mod in deprecated:
        for suffix in ("weight", "bias"):
            key = f"{mod}.{suffix}"
            if key in state_dict:
                print(f"WARNING: Removing deprecated key {key} from model")
                state_dict.pop(key)
    model.load_state_dict(state_dict)
    model_noise.load_state_dict(state_dict)

    return model, data_loaders, model_noise

def evaluation(config, logger=None, trainer=None):

    if getattr(config._args, "per_class", False):
        name_test_txt = config._args.config.split('configs/audiocaps/train-vggish-vggsound-')[1].split('.json')[0]
        name_test_txt = f"{name_test_txt}.txt"
        with open(Path('data/AudioCaps/structured-symlinks') / name_test_txt, 'r') as f:
            relevant_ids = f.read().splitlines()
    
    if logger is None:
        logger = config.get_logger('test')

    if getattr(config._args, "eval_from_training_config", False):
        eval_conf = copy.deepcopy(config)
        merge(eval_conf._config, config["eval_settings"], strategy=Strategy.REPLACE)
        config = eval_conf

    logger.info("Running evaluation with configuration:")
    logger.info(config)

    # Set the random initial seeds
    seed = config["seed"]
    logger.info(f"Setting experiment random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    # prepare model for testing.  Note that some datasets fail to fit the retrieval
    # set on the GPU, so we run them on the CPU
    if torch.cuda.is_available() and not config.get("disable_gpu", True):
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Running evaluation on {device}")

    model, data_loaders, model_noise = get_model_and_data_loaders(
        config=config,
        logger=logger,
        ckpt_path=Path(config._args.resume),
        device=device
    )
    logger.info(model)

    update_src_web_video_dir(config)
    visualizer = config.init(
        name='visualizer',
        module=module_vis,
        exp_name=config._exper_name,
        web_dir=config._web_log_dir,
    )

    metrics = [getattr(module_metric, met) for met in config['metrics']]
    challenge_mode = config.get("challenge_mode", False)
    challenge_msg = (
        "\n"
        "Evaluation ran on challenge features. To obtain a score, upload the similarity"
        "matrix for each dataset to the test server after running the "
        "`misc/cvpr2020-challenge/prepare_submission.py` script and following the "
        "instructions at: "
        "https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/"
        "\n"
    )

    model = model.to(device)
    model.eval()


    with torch.no_grad():
        samples, meta = data_loaders["retrieval"]
        shape_mask = meta['query_masks'].shape
        if getattr(config._args, "per_class", False):
            video_names = meta['paths']
            video_names = [video_name.split('videos/')[1].split('.mp4')[0] for video_name in video_names]
            query_masks_class_v2t = np.zeros(len(meta['query_masks']))
            query_masks_class_t2v = np.zeros(shape_mask)
            for idx, video_name in enumerate(video_names):
                if video_name in relevant_ids:
                    query_masks_class_t2v[idx]=np.ones((1, shape_mask[1]))
                    query_masks_class_v2t[idx]=np.array([1.])
        else:
            query_masks_class_t2v = None
            query_masks_class_v2t = None

        # To use the nan-checks safely, we need make temporary copies of the data
        disable_nan_checks = config._config["disable_nan_checks"]
        with ctxt_mgr(samples, device, disable_nan_checks) as valid:
            output = model(**valid)

        reg_sims = output["cross_view_conf_matrix"].data.cpu().float().numpy()
        dataset = data_loaders.dataset_name

        for fusion_name in ["combsum","rrf","weighted_combsum"]:
            print(fusion_name)
            if fusion_name=="combsum":
                sims = combsum(reg_sims)
            elif fusion_name=="rrf":
                sims= RRF(reg_sims)
            else:
                sims = pd.read_csv("data/weighted_combsum.csv").to_numpy()
            nested_metrics = {}

            for metric in metrics:
                metric_name = metric.__name__
                res = metric(sims, query_masks=None,
                             query_masks_class_t2v=query_masks_class_t2v,
                             query_masks_class_v2t=query_masks_class_v2t)
                verbose(epoch=0, metrics=res, name=dataset, mode=metric_name)
                if trainer is not None:
                    if not trainer.mini_train:
                        trainer.writer.set_step(step=0, mode="val")
                    # avoid tensboard folding by prefixing
                    metric_name_ = f"test_{metric_name}"
                    trainer.log_metrics(res, metric_name=metric_name_, mode="val")
                nested_metrics[metric_name] = res

            if data_loaders.num_test_captions == 1:
                visualizer.visualize_ranking(
                    sims=sims,
                    meta=meta,
                    epoch=0,
                    nested_metrics=nested_metrics,
                )
            log = {}
            for subkey, subval in nested_metrics.items():
                for subsubkey, subsubval in subval.items():
                    log[f"test_{subkey}_{subsubkey}"] = subsubval
            for key, value in log.items():
                logger.info(" {:15s}: {}".format(str(key), value))


def combsum(sims, N=5):
    return sims.reshape(-1, N, sims.shape[-1]).sum(axis=1)


def RRF(sims, v=60):
    ranking = np.argsort(np.argsort(sims, axis=1), axis=1)
    ranking = -1 * (ranking - ranking.shape[1])# make rankings in ascending order (best -> worst)
    scores = np.apply_along_axis(lambda x: 1 / (x + v), 1, ranking)
    return combsum(scores)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default=None, type=str, help="config file path")
    args.add_argument('--resume', type=Path, help='path to checkpoint for evaluation')
    args.add_argument('--device', help='indices of GPUs to enable')
    args.add_argument('--eval_from_training_config', action="store_true",
                      help="if true, evaluate directly from a training config file.")
    args.add_argument("--custom_args", help="qualified key,val pairs")
    args.add_argument("--per_class", action="store_true",
                      help="if true, evaluate retrieval task only on specific class")
    eval_config = ConfigParser(args)

    cfg_msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert eval_config._args.resume, cfg_msg
    evaluation(eval_config)
