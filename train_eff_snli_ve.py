'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from eff_blip.eff_blip_snli_ve import blip_snli_ve
import utils
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result

from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

def train(model, data_loader, optimizer, scheduler, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    for i,(image, hypothesis, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)

        loss = model(image, hypothesis, label)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # if i == 51:
        #     break
        # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate SNLI_VE test result:'
    print_freq = 200
    
    result = []

    answer_list = data_loader.dataset.answer_list
    answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
    answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id

    for n, (image, hypothesis, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        answer_ids = model.rank_answer(image, hypothesis, answer_list, answer_candidates, k=config['k_test'])

        for l, answer_id in zip(label, answer_ids):
            result.append({"label": l, "answer": answer_list[answer_id]})

    return result

def cal_snli_ve_score(result, result_dir, epoch, split):
    result_file = os.path.join(result_dir, f'temp_{split}_result_rank{utils.get_rank()}.json')
    json.dump(result, open(result_file, 'w'))

    final_result_file = os.path.join(result_dir, f'{split}_epoch{epoch}.json')

    score = 0

    dist.barrier()

    if utils.is_main_process():
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, f'temp_{split}_result_rank{rank}.json')
            res = json.load(open(result_file, 'r'))
            result += res

            # delete temp val file
            os.remove(result_file)

        for r in result:
            if r['label'] == r['answer']:
                score += 1

        score /= len(result)

        json.dump(result,open(final_result_file,'w'))
        print('result file saved to %s'%final_result_file)

    return score


def main(args, config):
    utils.init_distributed_mode(args)

    start_epoch = config['start_epoch']
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(seed)

    #### Dataset ####
    print("Creating snli_ve datasets")
    datasets = create_dataset('snli_ve', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[2, 2, 2], is_trains=[True, False, False], collate_fns=[None, None, None])

    #### Model #### 
    print("Creating model")

    if not args.evaluate:
        model = blip_snli_ve(pretrained=config['pretrained'], config=config, image_size=config['image_size'],
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                         vit_ckpt_layer=config['vit_ckpt_layer'])
    else:
        model = blip_snli_ve(pretrained=config['trained'], config=config, image_size=config['image_size'],
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                         vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)
       
    freeze_list = ["adapter"]

    if config['freeze']:
        for name, param in model.named_parameters():
            flag = 0
            for f in freeze_list:
                if f in name:
                    flag = 1
                    break
            if flag == 0:
                param.requires_grad = False

        if utils.is_main_process():
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    print(name)

    dist.barrier()
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module    

    t_step = len(train_loader) * config['max_epoch']
    w_step = int(t_step * config['warmup_ratio'])

    optimizer = AdamW(params=model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, w_step, t_step)

    best = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, scheduler, epoch, device)

            dist.barrier()

            if args.offline_val:

                snli_ve_val_result = evaluation(model_without_ddp, val_loader, device, config)
                val_acc = cal_snli_ve_score(snli_ve_val_result, args.result_dir, epoch, split='val')

                snli_ve_test_result = evaluation(model_without_ddp, test_loader, device, config)
                test_acc = cal_snli_ve_score(snli_ve_test_result, args.result_dir, epoch, split='test')

                if utils.is_main_process():

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 'epoch': epoch,
                                 }

                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'test_acc': test_acc,
                        'val_acc': val_acc,
                        'best_acc': best
                    }

                    if val_acc >= best:
                        print(f"snli_ve score improve from {best} --------> {val_acc}, test score: {test_acc}")
                        best = val_acc
                        save_obj['best_acc'] = best

                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_last.pth'))
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch
                }

                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_last.pth'))
        else:
            break

        dist.barrier()
                      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./eff_blip_configs/eff_snli_ve.yaml')
    parser.add_argument('--output_dir', default='./output/snli_ve/')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--offline_val', default=True, type=bool)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)