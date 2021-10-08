import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mergedeep import Strategy, merge
from typeguard import typechecked

import model.metric as module_metric
from trainer import verbose
from utils.util import update_src_web_video_dir
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



def evaluation(config, logger=None, trainer=None):

    if logger is None:
        logger = config.get_logger('test')

    if getattr(config._args, "eval_from_training_config", False):
        eval_conf = copy.deepcopy(config)
        merge(eval_conf._config, config["eval_settings"], strategy=Strategy.REPLACE)
        config = eval_conf

    logger.info("Running evaluation with configuration:")
    logger.info(config)

    # Set the random initial seeds
    seed = config["seed"] #0
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

    update_src_web_video_dir(config)


    metrics = [getattr(module_metric, met) for met in config['metrics']]

    with torch.no_grad():
        query_masks_class_t2v = None
        query_masks_class_v2t = None


        reg_sims = pd.read_csv("data\sim_mat_test.csv", dtype=float).to_numpy()#output["cross_view_conf_matrix"].data.cpu().float().numpy()
        dataset = 'CLOTHO' #data_loaders.dataset_name

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
    evaluation(eval_config)
