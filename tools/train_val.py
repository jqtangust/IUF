import argparse
import logging
import os
import pprint
import shutil
import time

import torch
import torch.distributed as dist
import torch.optim
import torch.nn as nn

import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.criterion_helper import build_criterion, build_svd_loss
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import visualize_compound, visualize_single
from functools import partial

device = torch.device("cuda")  # 选择CUDA设备

parser = argparse.ArgumentParser(description="UniAD Framework")
parser.add_argument("--config", default=None)
parser.add_argument("-e", "--evaluate", default=True, action="store_true")
parser.add_argument("--local_rank", default=0, help="local rank for dist")


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    alpha = alpha
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ema_model_state_dict = ema_model.state_dict()

        for entry in model_state_dict.keys():
            ema_param = ema_model_state_dict[entry].clone().detach()
            param = model_state_dict[entry].clone().detach()

            # Interpolation
            new_param = (ema_param * alpha) + (param * (1. - alpha))

            model_state_dict[entry] = new_param

        model.load_state_dict(model_state_dict)


def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    rank = 0
    config = update_config(config)

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(
        config.exp_path, config.evaluator.save_dir)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(
            config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path +
            "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = ModelHelper(config.net)
    model = nn.DataParallel(model)
    model.module.cuda()
    model.cuda()
    
    # create model_old
    if config.trainer.optimizer.type == "ConstrainedSGD":
        print("Activate Old Model")
        model_old = ModelHelper(config.net)
        model_old = nn.DataParallel(model_old)
        model_old.module.cuda()
        model_old.cuda()
        skip = False
    else:
        model_old = None
        skip = True

    local_rank = int(args.local_rank)

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    # parameters needed to be updated
    parameters = [
        {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, model, config.trainer.optimizer)

    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    # load model: auto_resume > resume_model > load_path
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)

    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")

    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model

    if resume_model:
        best_metric, last_epoch = load_state(
            resume_model, model, optimizer=optimizer)
        best_metric, last_epoch = load_state(
            resume_model, model_old, optimizer=optimizer)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)
        if skip == False:   
            load_state(load_path, model_old)

    # Dataloader
    train_loader, train_val_loader, val_loader = build_dataloader(
        config.dataset, distributed=False)

    ############################################################
    # if only need to test, uncomment the following code
    
    # if args.evaluate:
    #     if resume_model:
    #         ret_metrics, outputs_dict = validate(train_val_loader, val_loader, model ,1, ori_feature=torch.load(resume_model)["feature_metrix"])
    #     else:
    #         ret_metrics, outputs_dict = validate(train_val_loader, val_loader, model ,1, ori_feature={})
    #     return
    ############################################################
    
    if config.trainer.optimizer.type == "ConstrainedSGD":
        print("save feature metrix")
        optimizer.add_feature_matrix(feature_metrix_cal(torch.load(resume_model)["feature_metrix"].copy()),logger)

    criterion = build_criterion(config.criterion)
    CELoss = nn.CrossEntropyLoss()
    SVDLoss = build_svd_loss()


    for epoch in range(last_epoch, config.trainer.max_epoch):
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            train_loader,
            model,
            model_old,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            CELoss,
            SVDLoss,
            frozen_layers,
            skip
        )
        # return 0
        lr_scheduler.step()
        torch.cuda.empty_cache()

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            if resume_model:
                ret_metrics, outputs_dict = validate(train_val_loader, val_loader, model ,epoch + 1, ori_feature=torch.load(resume_model)["feature_metrix"])
            else:
                ret_metrics, outputs_dict = validate(train_val_loader, val_loader, model ,epoch + 1, ori_feature={})
            if rank == 0:
                ret_key_metric = ret_metrics[key_metric]
                is_best = ret_key_metric >= best_metric
                best_metric = max(ret_key_metric, best_metric)
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": config.net,
                        "state_dict": model.state_dict(),
                        "best_metric": best_metric,
                        "optimizer": optimizer.state_dict(),
                        "feature_metrix": outputs_dict,
                    },
                    is_best,
                    config,
                )

def train_one_epoch(
    train_loader,
    model,
    model_old,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    CELoss,
    SVDLoss,
    frozen_layers,
    skip
):

    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    
    losses = AverageMeter(config.trainer.print_freq_step)
    losses_class = AverageMeter(config.trainer.print_freq_step)
    losses_svd = AverageMeter(config.trainer.print_freq_step)

    model.train()

    # freeze selected layers
    for layer in frozen_layers:
        module = getattr(model.module, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    rank = 0
    logger = logging.getLogger("global_logger")
    end = time.time()

    concatenated_tensor_0 = torch.empty(0).cuda()
    concatenated_tensor_1 = torch.empty(0).cuda()
    concatenated_tensor_2 = torch.empty(0).cuda()

    for i, input in enumerate(train_loader):
        
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # interpolation
        if skip == False:
            update_ema_variables(model, model_old, 0.05, curr_step)

        # forward
        outputs = model(input)
        
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)

        concatenated_tensor_0 = torch.cat([concatenated_tensor_0, outputs["middle_decoder_feature_0"].clone().detach()], dim=0)
        concatenated_tensor_1 = torch.cat([concatenated_tensor_1, outputs["middle_decoder_feature_1"].clone().detach()], dim=0)
        concatenated_tensor_2 = torch.cat([concatenated_tensor_2, outputs["middle_decoder_feature_2"].clone().detach()], dim=0)
        
        if concatenated_tensor_0.shape[0] > 768:
            concatenated_tensor_0 = concatenated_tensor_0[48:]
        
        if concatenated_tensor_1.shape[0] > 768:
            concatenated_tensor_1 = concatenated_tensor_1[48:]
        
        if concatenated_tensor_2.shape[0] > 768:
            concatenated_tensor_2 = concatenated_tensor_2[48:]
        
        loss_svd = 0
        loss_svd = SVDLoss(concatenated_tensor_0,concatenated_tensor_1,concatenated_tensor_2)

        input["clslabel"] = input["clslabel"].to(device)  # 将张量移动到CUDA设备上
        class_loss = CELoss(outputs["class_out"], input["clslabel"])

        loss += class_loss
        if skip == True:
            loss += 10*loss_svd

        reduced_loss = loss.clone()
        reduced_loss_class = class_loss.clone()
        reduced_loss_svd = loss_svd.clone()

        losses.update(reduced_loss.item())
        losses_class.update(reduced_loss_class.item())
        losses_svd.update(reduced_loss_svd.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("svd_loss", losses_svd.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)

            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                # "Loss_Now {loss_now.val:.5f} ({loss_now.avg:.5f})\t"
                "Loss_Class {loss_class.val:.5f} ({loss_class.avg:.5f})\t"
                "Loss_SVD {loss_svd.val:.5f} ({loss_svd.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_class=losses_class,
                    loss_svd=losses_svd,
                    lr=current_lr,
                )
            )
        end = time.time()
              
# 定义钩子函数
def hook_fn(module, input, output, outputs_dict, name):
    # 保存每一层的输出
    module_name = str(name)
    if torch.is_tensor(output):
        device = torch.device("cuda:0")
        output = output.mean(dim=0).to(device) 
        if module_name in outputs_dict:
            outputs_dict[module_name] = torch.cat(
                [outputs_dict[module_name], output.detach()], dim=0)
        else:
            outputs_dict[module_name] = output.detach()

def feature_metrix_cal(outputs_dict):
    feature_metrix_dict={}
    for k,v in outputs_dict.items():
        U, S, Vh = torch.linalg.svd(
            v, full_matrices=False)
        feature_metrix_dict[k] = Vh
    return feature_metrix_dict

def validate(train_val_loader, val_loader, model, epoch, ori_feature):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    # rank = dist.get_rank()
    rank = 0
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
   
    outputs_dict = {}

    # Memory Dict
    feature_metrix_dict={}
    print(epoch)
    
    merged_dict = ori_feature
    if epoch % 2000 == 0:    
        for name, layer in model.named_modules():
            if ("transformer" in name and "decode" in name and "linear2" in name) or ("transformer" in name and "encoder" in name and "linear2" in name):
                # print("LAYER:", name)
                handle = layer.register_forward_hook(
                    partial(hook_fn, outputs_dict=outputs_dict, name=name))
            
        with torch.no_grad():
            for i, input in enumerate(train_val_loader):
                # forward
                logger.info(i)
                outputs = model(input)

            if ori_feature:
                merged_dict = {k: torch.cat([ori_feature[k], outputs_dict[k]], dim=0) for k in outputs_dict.copy()}
            else:
                merged_dict = outputs_dict.copy()
            
            handle.remove()

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input)

            dump(config.evaluator.eval_dir, outputs, input)

            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(input["filename"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

    # gather final results
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {}  # only ret_metrics on rank0 is not empty

    if rank == 0:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(
            " * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
        shutil.rmtree(config.evaluator.eval_dir)
        # evaluate, log & vis
        ret_metrics = performances(
            fileinfos, preds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)
        if args.evaluate and config.evaluator.get("vis_compound", None):
            logger.info("DO vis_compound")
            visualize_compound(
                fileinfos,
                preds,
                masks,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
            )
        if args.evaluate and config.evaluator.get("vis_single", None):
            logger.info("DO vis_single")
            visualize_single(
                fileinfos,
                preds,
                config.evaluator.vis_single,
                config.dataset.image_reader,
            )
    model.train()
    return ret_metrics, merged_dict


if __name__ == "__main__":
    main()
