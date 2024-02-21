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

from eff_blip.eff_blip_vqa import blip_vqa
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
    
    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      

        loss = model(image, question, answer, n=n, weights=weights)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # if i == 51:
        #     break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 200
    
    result = []

    if config['inference'] == 'rank':
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
        answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)

        if config['inference'] == 'generate':
            answers = model.generate(image, question, sample=False, num_beams=config['num_beams'],
                                     max_length=config['max_length'],
                                     min_length=config['min_length'])

            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())
                result.append({"question_id": ques_id, "answer": answer})

        elif config['inference'] == 'rank':
            answer_ids = model.rank_answer(image, question, answer_list, answer_candidates, k=config['k_test'])

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id": int(ques_id.item()), "answer": answer_list[answer_id]})

        # if n == 10:
        #     break
        # break

    return result

def cal_vqa_score(result, ann_root, result_dir, epoch, dataset, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json' % ('temp_val_result', utils.get_rank()))
    json.dump(result, open(result_file, 'w'))

    final_result_file = os.path.join(result_dir, f'{dataset}_epoch%d.json' %epoch)

    score = 0

    dist.barrier()

    if utils.is_main_process():
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json' % ('temp_val_result', rank))
            res = json.load(open(result_file, 'r'))
            result += res

            # delete temp val file
            os.remove(result_file)

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new


        annotation = json.load(open(os.path.join(ann_root, f'{dataset}.json'), 'r'))
        annotation_dict = {}
        for a in annotation:
            answers_dict = {}
            for answer in set(a['answer']):
                answer_count = a['answer'].count(answer)
                answers_dict[answer] = min(np.float32(answer_count) * 0.3, 1)

            annotation_dict[a['question_id']] = answers_dict

        for r in result:
            ans = r['answer']
            ans_dict = annotation_dict[r['question_id']]
            if ans in ans_dict.keys():
                score += ans_dict[ans]
            else:
                score += 0

        score /= len(result)

        json.dump(result, open(final_result_file,'w'))
        print('result file saved to %s'%final_result_file)

    return score


def main(args, config):
    utils.init_distributed_mode(args)

    start_epoch = 0
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[4, 4, 4],is_trains=[True, False, False],
                                              collate_fns=[vqa_collate_fn, None, None])

    #### Model #### 
    print("Creating model")

    if not args.evaluate:
        model = blip_vqa(pretrained=config['pretrained'], config=config, image_size=config['image_size'],
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                         vit_ckpt_layer=config['vit_ckpt_layer'])
    else:
        model = blip_vqa(pretrained=config['trained'], config=config, image_size=config['image_size'],
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                         vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)

    # resume training
    if start_epoch > 0:
        resume_state_dict = torch.load(config['trained'], map_location='cpu')
        print(f"resume from {config['trained']}")

        model.load_state_dict(resume_state_dict['model'])
        
    # for name, param in model.named_parameters():
    #     if "visual_encoder" in name:
    #         param.requires_grad = False
    
    freeze_list = ["prefix", "adapter"]

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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    

    t_step = len(train_loader) * config['max_epoch']
    w_step = int(t_step * config['warmup_ratio'])

    optimizer = AdamW(params=model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, w_step, t_step)

    best = 0

    if start_epoch > 0:
        optimizer.load_state_dict(resume_state_dict['optimizer'])
        scheduler.load_state_dict(resume_state_dict['scheduler'])
        best = resume_state_dict['val_acc']
       
    print("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            train_stats = train(model, train_loader, optimizer, scheduler, epoch, device)

            dist.barrier()

            if args.offline_val:

                vqa_val_result = evaluation(model_without_ddp, val_loader, device, config)
                val_acc = cal_vqa_score(vqa_val_result, config['ann_root'], args.result_dir, epoch, 'vqa_karpathy_val')

                vqa_test_result = evaluation(model_without_ddp, test_loader, device, config)
                test_acc = cal_vqa_score(vqa_test_result, config['ann_root'], args.result_dir, epoch, 'vqa_karpathy_test')

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
                        print(f"vqa score improve from {best} --------> {val_acc}, test score: {test_acc}")
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

    if args.evaluate:
        vqa_result = evaluation(model_without_ddp, test_loader, device, config)
        _ = save_result(vqa_result, args.result_dir, 'vqa_result')
                      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./eff_blip_configs/eff_vqa.yaml')
    parser.add_argument('--output_dir', default='/research/d4/gds/xyzhang21/AttnAdapter/output/vqa_l_p')
    # parser.add_argument('--output_dir', default='/research/d4/gds/xyzhang21/AttnAdapter/output/vqa_adapter')
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