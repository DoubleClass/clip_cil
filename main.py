
import os
import random
import json
import numpy as np
import hydra
import logging
from omegaconf import DictConfig
from tqdm import tqdm

import torch
import statistics
from torch.utils.data import DataLoader
from continuum.metrics import Logger
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import clip
from clip_base import utils
from clip_base.models import ClassIncremental
from clip_base.models import sample


from clip_base.datasets import build_cl_scenarios
from continuum import rehearsal
from copy import deepcopy


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def _get_optimizer(model, cfg):
    """Returns the optimizer"""
   
    params = model.clip_mapping.parameters()
    return torch.optim.Adam(params, lr=0.001)

def cross_entropy(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    """Calculates cross-entropy with temperature scaling"""
    out = torch.nn.functional.softmax(outputs, dim=1)
    tar = torch.nn.functional.softmax(targets, dim=1)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce

def get_mixed_clip(model, reuse):
    clip_mapping = model.clip_mapping.parameters()
    new_weight = next(clip_mapping)
    old_weight = model.old_clip.fc.weight
    diff = old_weight.sub(new_weight)
    diff = torch.abs(diff)
    va, ind = torch.topk(diff.view(-1).abs(), int(512*512*reuse))
    final_weight = torch.where(diff < va[-1], old_weight, new_weight)
    model.clip_mapping.fc.weight.data = final_weight

def get_mixed_clip_random(model, reuse, device):
    clip_mapping = model.clip_mapping.parameters()
    new_weight = next(clip_mapping)
    old_weight = model.old_clip.fc.weight
    final_weight = deepcopy(old_weight)
    pair_set = set()
    dim = (512, 512)
    mask = torch.zeros((512, 512))
    num_keep = int(np.prod(dim) * reuse)
    mask_indices = torch.randperm(np.prod(dim))[:num_keep]
    mask.view(-1)[mask_indices] = 1
    mask = mask.to(device)
    final_weight = mask * old_weight + (1 - mask) * new_weight
    model.clip_mapping.fc.weight.data = final_weight



def get_logger(filename):
    logger = logging.getLogger(name='r')  # 不加名称设置root logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s- %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # 添加两个Handler
    # logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    
    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)
    utils.save_config(cfg)

    logging_filename = os.path.join(os.getcwd(), "my.log")
    my_logger = get_logger(logging_filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    _, transforms = clip.load(cfg.model_name, device=device)
    train_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=transforms
    )
    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=transforms
    )
    new_class_name = utils.get_ordered_class_name(cfg.class_order, classes_names)
    model = ClassIncremental(cfg, device, new_class_name)

    if 'mem_size' in cfg:
        mem_size = cfg.mem_size
        total_class = 1000
    elif cfg.dataset == 'imagenet_R':
        # import pdb; pdb.set_trace()
        total_class = 200
        mem_size = 1000
    else:
        # import pdb; pdb.set_trace()
        mem_size = 2000
        total_class = 100
    
    memory = rehearsal.RehearsalMemory(
        memory_size=mem_size,
        herding_method="barycenter",
        fixed_memory=False,
        nb_total_classes=total_class
    )
    


    metric_logger = Logger(list_subsets=["train", "test"])

    acc_list = []

    if 'total_class' in cfg:
        total_task = (cfg.total_class-cfg.initial_increment)/cfg.increment
    elif cfg.dataset == "imagenet_R":
        total_task = 9
    else:
        total_task = (100-cfg.initial_increment)/cfg.increment
    # import pdb; pdb.set_trace()
    ## get config
    reuse = cfg.reuse

    for task_id, taskset in enumerate(train_dataset):
        optm = _get_optimizer(model, cfg)
        total_epoch = cfg.EPOCH
        if task_id == 0:
            scheduler1 = MultiStepLR(optm, milestones=[4,10], gamma=0.1)
        else:
            scheduler1 = CosineAnnealingLR(optm, T_max=total_epoch)
       
        my_logger.info(f"Train for task {task_id} has started.")
        # get dataloader contains memory
        loader = DataLoader(taskset, shuffle=True, batch_size=cfg.batch_size, num_workers=8)
        model.train()
        if not cfg.zero:
            for epoch in range(total_epoch):
                batch_id = -1
                if task_id >0:
                    random_class_order_list = list(range(cfg.initial_increment+(task_id-1)*cfg.increment))
                    random.shuffle(random_class_order_list)
                right = 0
                total = 0
                idx= 0
                for inputs, targets, task_ids in tqdm(loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    batch_id += 1
                    memory_inputs = None
                    if cfg.examplar_free and task_id > 0:
                        memory_inputs = []
                        memory_targets = []
                        list_for_one_batch = [random_class_order_list[batch_id*2%len(random_class_order_list)], random_class_order_list[(batch_id*2+1)%len(random_class_order_list)]]
                        if cfg.dataset == "cifar100" and cfg.increment == 5:
                            list_for_one_batch = [random_class_order_list[batch_id*4%len(random_class_order_list)], random_class_order_list[(batch_id*4+1)%len(random_class_order_list)], random_class_order_list[(batch_id*4+2)%len(random_class_order_list)], random_class_order_list[(batch_id*4+3)%len(random_class_order_list)]]
                        for i in list_for_one_batch:
                            memory_inputs.append(sample(model.class_mean_list[i], model.class_cov_list[i],int(10*cfg.beta), shrink=cfg.shrinkage))
                            memory_targets.append(torch.ones(int(10*cfg.beta), dtype=torch.long, device=device)*i)
                        memory_inputs = torch.cat(memory_inputs, dim=0)
                        memory_targets = torch.cat(memory_targets, dim=0)

                        targets = torch.cat([targets, memory_targets], dim=0)
                        
                    idx += 1
                    outputs,fea = model(inputs, is_train=True, task_id=task_id, memory_data=memory_inputs)

                    right += np.sum(list(outputs.cpu().argmax(dim=1)==targets.cpu()))
                    if memory_inputs is not None:
                        total += inputs.shape[0] + memory_inputs.shape[0]
                    else:
                        total += inputs.shape[0]
                    
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                    optm.zero_grad()
                    loss.backward()
                    optm.step()
                my_logger.info(f"-    - curent task {task_id},  epoch { epoch} acc: {right/total*100} ")
                scheduler1.step()
               

                if epoch == total_epoch - 1 and task_id > 0 and reuse != 0:
                    my_logger.info(f"the last epoch, starting changing cilp mapping")
                    if cfg.reuse:
                        if cfg.random_reuse:
                            get_mixed_clip_random(model, reuse, device=device)
                        else:
                            get_mixed_clip(model, reuse)
            my_logger.info(f"finished task: {task_id,} starting eval...")
        
        if cfg.examplar_free:
            print('generating examplar...')
            examplar_loader = DataLoader(train_dataset[task_id], batch_size=cfg.examplar_batch, shuffle=False, num_workers=cfg.num_workers)
            examplar_data = []
            examplar_target = []
            examplar_after_adapt_feature = []
            for input, target, task_ids in tqdm(examplar_loader):
                input, target = input.to(device), target.to(device)
                with torch.no_grad():
                    _, ori_ima_feat = model(input, ori_ima_f=True)
                examplar_data.append(ori_ima_feat)
                examplar_target.append(target)
            examplar_target = torch.cat(examplar_target, dim=0)
            examplar_data = torch.cat(examplar_data, dim=0)
            model.analyze_mean_cov(examplar_data, examplar_target)
            print('done')


        # eval 
        model.post_train(task_id, cfg.increment)
        model.eval()
        a_shadow = 0
        a_right = 0
        a_long = 0
        with torch.no_grad():
            eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=128)
            right = 0 
            total = 0
            for inputs, targets, task_ids in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs, _ = model(inputs, False, task_id, targets)
               
                right += np.sum(list(outputs.cpu().argmax(dim=1)==targets.cpu()))
                total += inputs.shape[0]
                metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
            acc_list.append(100 * metric_logger.accuracy)
            my_logger.info(f"test on seen classes: acc: {metric_logger.accuracy}")
            with open(cfg.log_path, 'a+') as f:
                f.write(json.dumps({
                    'task': task_id,
                    'acc': round(100 * metric_logger.accuracy, 2),
                    'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                    'forgetting': round(100 * metric_logger.forgetting, 6),
                    'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                    'bwt': round(100 * metric_logger.backward_transfer, 2),
                    'fwt': round(100 * metric_logger.forward_transfer, 2),
                }) + '\n')
                metric_logger.end_task()
        if task_id > 0:
            my_logger.info(f'right: {a_right} shadow: {a_shadow},  long: {a_long}')
        # prepare exemplars
        if task_id != total_task:
            if not cfg.linear_probe and not cfg.task_cls and not cfg.task_inc:
                model.get_old_model()    
            model.adaptation(task_id, cfg.increment)

    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')


if __name__ == "__main__":
    torch.set_num_threads(32)
    continual_clip()
