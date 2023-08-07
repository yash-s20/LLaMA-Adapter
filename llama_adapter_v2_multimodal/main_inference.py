import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from llama.llama_adapter import LLaMA_adapter

from data.dataset import TestDataset, transform_train

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

from engine_inference import test_all_data


def get_args_parser():
    parser = argparse.ArgumentParser('llama_adapterV2 pre-training', add_help=False)

    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--max_words', default=96, type=int,
                        help='max number of input words')
    parser.add_argument("--checkpoint_epoch", type=int, help="Which checkpoint to load for testing")

    # Dataset parameters
    parser.add_argument('--data_config', default='configs/data/pretrain/EN.yaml', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--split_epoch', type=int, default=1)
    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print("", flush=True)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False

    # define the model
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    print("Initializing model")
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path, phase="pretrain")
    print("Model initialized")
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print("Trainable Params:")
    print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    # training detail
    eff_batch_size = 1 # args.batch_size * args.accum_iter * misc.get_world_size()

    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers



    dataset_test = TestDataset(args.data_config, transform=transform_train,
                                max_words=args.max_words, tokenizer_path=llama_tokenzier_path)
    print(dataset_test)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = misc.DistributedSubEpochSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, split_epoch=args.split_epoch, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_test = iter(dataset_test)

    # SummaryWrite
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    if args.checkpoint_epoch > 0:
        load_model_path = os.path.join(args.output_dir, 'checkpoint-%s.pth' % (args.checkpoint_epoch - 1 ))
        print(f"Loading {load_model_path}")
        misc.load_model(model_without_ddp, load_model_path)
    start_time = time.time()
    data_loader_test.sampler.set_epoch(0)

    train_stats = test_all_data(
        model, data_loader_test,
        device, 0,
        log_writer, args=args
    )

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': 0}

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
