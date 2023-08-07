import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from llama import LLaMA_adapter

def test_all_data(model: LLaMA_adapter, data_loader: Iterable,
                    device: torch.device, epoch: int,
                    log_writer=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (input_text, img) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        print(img)
        print(input_text)
        text = input_text
        img = img.to(device, non_blocking=True)
        img = model.clip_transform(img).unsqueeze(0).to(device)
        output = model.generate(img, text)
        if device == "cuda":
            torch.cuda.synchronize()
        print(output)
        
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        # write accuracy

        # log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
        # log_writer.add_scalar('m_train_loss', m_loss_value_reduce, epoch_1000x)
        # log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
