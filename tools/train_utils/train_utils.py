import glob
import os

import numpy as np

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

from tools.eval_utils.ap_helper import APCalculator, parse_predictions, parse_groundtruths

def train_one_epoch(model, optimizer, train_loader, val_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    
    train_running_loss = 0.0
    val_running_loss = 0.0

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(len(train_loader)):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()
        loss, tb_dict, disp_dict = model_func(model, batch) # try print it
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        train_running_loss += loss.item()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        if cur_it % 47 == 46:
        #if cur_it % 50 == 49:
            if True:
            #val_dataloader_iter = iter(val_loader)
            #with torch.no_grad():
                #model.eval()
                #for i in range(len(val_loader)):
                    #try:
                        #batch = next(val_dataloader_iter)
                    #except StopIteration:
                        #val_dataloader_iter = iter(val_loader)
                        #batch = next(val_dataloader_iter)
                        #print('new iters for validation')
                    #loss, _, _ = model_func(model, batch) # try print it
                    #val_running_loss += loss.item()
                train_running_loss /= 47
                #val_running_loss /= (i+1)

                 # log to console and tensorboard
                if rank == 0:
                    pbar.update()
                    pbar.set_postfix(dict(total_it=accumulated_iter))
                    tbar.set_postfix(disp_dict)
                    tbar.refresh()

                    if tb_log is not None:
                        tb_log.add_scalars('train',{'training_loss':train_running_loss,'val_loss':val_running_loss}, accumulated_iter)
                        tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                        for key, val in tb_dict.items():
                            tb_log.add_scalar('train/' + key, val, accumulated_iter)
            train_running_loss = 0
            val_running_loss = 0

    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_model(model, optimizer, train_loader,val_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, dataset="3rscan"):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, val_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            if cur_epoch % 10 == 9:
            #if True:
                val_dataloader_iter = iter(val_loader)
                np.set_printoptions(suppress=True)
                if dataset == "3rscan":
                    AP = np.zeros([2,8])
                    AP_counter = np.zeros([2,8])+1e-6
                else:
                    AP = np.zeros([2, 19])
                    AP_counter = np.zeros([2, 19]) + 1e-6
                AP_IOU_THRESHOLDS = [0.25, 0.5]
                ap_calculator_list = [APCalculator(iou_thresh) for iou_thresh in AP_IOU_THRESHOLDS]
                with torch.no_grad():
                    model.eval()
                    for i in range(len(val_loader)):
                        try:
                            batch = next(val_dataloader_iter)
                        except StopIteration:
                            val_dataloader_iter = iter(val_loader)
                            batch = next(val_dataloader_iter)
                            print('new iters for validation')
                        _, pred_dicts, _ = model_func(model, batch) # try print it
                        batch_pred_map_cls = parse_predictions(pred_dicts)
                        batch_gt_map_cls = parse_groundtruths(batch)
                        for ap_calculator in ap_calculator_list:
                            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
                        for i, ap_calculator in enumerate(ap_calculator_list):
                            #print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
                            metrics_dict = ap_calculator.compute_metrics()
                            #mAP[i] += metrics_dict['mAP']
                            del metrics_dict['mAP']
                            for key in metrics_dict:
                                AP[i,int(float(key))] += metrics_dict[key]
                                AP_counter[i,int(float(key))] += 1
                                #print('eval %s: %f'%(key, metrics_dict[key]))
                    print((AP/AP_counter*100)[:,1:])
                    print(np.mean((AP/(AP_counter) * 100)[:,1:],axis=1))

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
