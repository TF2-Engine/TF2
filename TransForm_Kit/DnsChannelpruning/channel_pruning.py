# Copyright 2019 Inspur Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import torch.nn as nn
import torch
import time
from parse_model import ParseModel
from find_dnscp_conv import *
from dnscp_core import DnscpCore
from utility import *
from model_convert import *
from finetune import validate
from finetune import train_val
''' dnscp train and val'''
def dnscp_train_val(train_loader, val_loader,model, InputShape,criterion, optimizer,args):
    """  find connect layer!  """
    print('find connect layer begin...')
    parse_model_instance = ParseModel()
    layername_type_dict,layername_lainjie_dict = parse_model_instance.parse_model_caffe\
                                                (model, InputShape, softmax = False)
    print('find connect layer Done.')

    """  find  conv layer that needs to be pruned !  """
    print('find pruned layer begin...')
    conv_dnscp_flag_list = find_dnscp_conv(model,layername_type_dict,layername_lainjie_dict)
    print('find pruned layer Done.')
    print('conv_dnscp_flag:',conv_dnscp_flag_list)
    print('train begin')
    
    '''init dnscp core'''
    net_dnscp = DnscpCore() 
    best_acc1 = 0
    train_cp_stop_flag=False
    is_best = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        
        '''switch to train mode'''
        model.train()
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            '''measure data loading time'''
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            '''forward dnscp layer'''
            iter = i+len(train_loader)*epoch
            conv_param_before_cplist,conv_param_cpmasklist = \
            net_dnscp.forward_dnscp_layer(model,conv_dnscp_flag_list,iter)
            '''model forward'''
            output = model(images)
            '''measure accuracy and record loss'''
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            ''' compute gradient and do SGD step'''
            optimizer.zero_grad() 
            loss.backward()
            '''replace weight with cp before'''
            conv_index = 0
            for module in model.modules():
                if isinstance(module,nn.Conv2d):
                    module.weight.data = conv_param_before_cplist[conv_index]
                    conv_index+=1
            optimizer.step()        
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0 or i % (len(train_loader)-1) == 0:
                progress.display(i)
                pruned_model = generate_prunedmodel(model,conv_dnscp_flag_list,conv_param_cpmasklist)
                convparam_kept,convflops_kept = calculate_compress_rate(model,pruned_model,InputShape)
                print("kept ratio:",convflops_kept)
            '''control train epoch'''
            '''if convflops_kept >= args.kept_ratio+0.2:
                train_cp_stop_flag = True
                if iter<10000:
                    print("please set kept ratio bigger than ",args.kept_ratio)
                break'''
            if iter % 2000 == 0:
                pruned_model = generate_prunedmodel(model,conv_dnscp_flag_list,conv_param_cpmasklist)
                convparam_kept,convflops_kept = calculate_compress_rate(model,pruned_model,InputShape)
                '''evaluate on validation set'''
                acc1 = dnscp_validate(val_loader, model, criterion, conv_param_cpmasklist,args)               
                if convflops_kept >= args.kept_ratio-0.05 and convflops_kept <= args.kept_ratio+0.05:
                    '''remember best acc@1 and save checkpoint'''
                    is_best = acc1 > best_acc1
                    best_acc1 = acc1
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'acc1': acc1,
                        'optimizer' : optimizer.state_dict(),
                        'mask':conv_param_cpmasklist,
                        'conv_cpflag':conv_dnscp_flag_list,
                        'kept_ratio':convflops_kept,
                        'iter':iter
                    }, is_best,args.snapshotmodelname)
                is_best = False               
        if train_cp_stop_flag == True:
            break 
def dnscp_train(train_loader, model, criterion, optimizer, epoch,net_dnscp,conv_dnscp_flag,InputShape,args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    train_cp_stop_flag=False
    '''switch to train mode'''
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        '''measure data loading time'''
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        '''compute output'''
        iter = i+len(train_loader)*epoch
        #print("iter:",iter)
        '''conv_index = 0
        for module in model.modules():
            if isinstance(module,nn.Conv2d):
                if conv_index == 2:
                    print("ith weight data:",module.weight.data.cpu().numpy()[0,0:16,:,:])
                conv_index+=1'''
        conv_param_before_cplist,conv_param_cpmasklist = net_dnscp.forward_dnscp_layer(model,conv_dnscp_flag,iter)

        '''conv_index = 0
        for module in model.modules():
            if isinstance(module,nn.Conv2d):
                if conv_index == 2:
                    print("ith mask weight data:",module.weight.data.cpu().numpy()[0,0:16,:,:])
                conv_index+=1'''
        output = model(images)
        loss = criterion(output, target)
        '''measure accuracy and record loss'''
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad() 
        loss.backward()
        '''replace weight with cp before'''
        conv_index = 0
        for module in model.modules():
            if isinstance(module,nn.Conv2d):
                '''if conv_index == 2:
                    print("optimizer i+1 weight data:",module.weight.data.cpu().numpy()[0,0:16,:,:])
                    print("grad:",module.weight.grad.cpu().numpy()[0,0:16,:,:])
                    print("before_update_mask weight data:",conv_param_before_cp[conv_index].cpu().numpy()[0,0:16,:,:])
                    print("before_update_mask weight data:",conv_param_before_cplist[conv_index].cpu().numpy()[0,0:16,:,:])'''
                module.weight.data = conv_param_before_cplist[conv_index]
                conv_index+=1
        optimizer.step()
        '''only print'''
        '''conv_index = 0
        for module in model.modules():
            if isinstance(module,nn.Conv2d):
                if conv_index == 2:
                    print("i+1 weight data:",module.weight.data.cpu().numpy()[0,0:16,:,:])
                    print("grad:",module.weight.grad.cpu().numpy()[0,0:16,:,:])
                    print("before_update_mask weight data:",conv_param_before_cplist[conv_index].cpu().numpy()[0,0:16,:,:])
                #module.weight.data = conv_param_before_cplist[conv_index]-args.lr*module.weight.grad
                conv_index+=1'''
                
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i % (len(train_loader)-1) == 0:
            progress.display(i)
            pruned_model = generate_prunedmodel(model,conv_dnscp_flag,conv_param_cpmasklist)
            convparam_kept,convflops_kept = calculate_compress_rate(model,pruned_model,InputShape)
            print("kept ratio:",convflops_kept)
            if convflops_kept >= args.kept_ratio+0.02:
                train_cp_stop_flag = True
                break
        #if i % (len(train_loader)//2) == 0 or i == len(train_loader):
            
        ###end simple test 
    return conv_param_cpmasklist,train_cp_stop_flag,convflops_kept

def dnscp_validate(val_loader, model, criterion, conv_param_cpmasklist,args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    '''replace weight with weight*mask for forward'''
    conv_index = 0
    conv_param_nocp_list = []
    for module in model.modules():
        if isinstance(module,nn.Conv2d):
            conv_param_nocp_list.append(module.weight.data)
            #print(module.weight.data.shape,conv_param_cpmasklist[conv_index].shape)
            module.weight.data = module.weight.data * conv_param_cpmasklist[conv_index]
            conv_index+=1
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        '''replace weight*mask with weight'''
        conv_index = 0
        for module in model.modules():
            if isinstance(module,nn.Conv2d):
                module.weight.data = conv_param_nocp_list[conv_index]
                conv_index+=1
    return top1.avg


