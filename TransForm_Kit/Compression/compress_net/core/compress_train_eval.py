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

import core.compress_core as compress_core
import os
import torch.nn as nn
import numpy as np
import torch
import shutil
import time
best_acc1 = 0

#generate mask with conv and fc param
def compress(model,args):
    if os.path.isfile(args.resume): #load from pre step model
        checkpoint = torch.load(args.resume)
        maskPre = checkpoint['mask']
		#conver maskPre to numpy for two power speed
        for index,mask in enumerate(maskPre):
            maskPre[index] = mask.cpu().numpy()        
    else:
        maskPre=[] #first step compress on src model
        for module in model.modules():
            if isinstance(module,nn.Conv2d) or isinstance(module,nn.Linear):
                mask = np.ones(module.weight.size())
                maskPre.append(mask)
                if module.bias is not None:
                    mask= np.ones(module.bias.size())
                    maskPre.append(mask)
    maskCur=[]
    icompressparam_index = 0
    two_power_filter_torch_list = [] # if not two_power ==0 else two_power
    pre_percentage = args.pre_percentage
    cur_percentage = args.cur_percentage
	
    for module in model.modules():
        if isinstance(module,nn.Conv2d) or isinstance(module,nn.Linear):
            print(module)
            torch_filterdata = module.weight.data
            #print(type(torch_filterdata),torch_filterdata.size(),torch_filterdata)
            np_filterdata = torch_filterdata.cpu().numpy()
            max_compresstum_exp_,min_compresstum_exp_ = compress_core.ComputeQuantumRange(np_filterdata,maskPre[icompressparam_index],7) 
            print('min and max:',max_compresstum_exp_,min_compresstum_exp_) 
            two_power_filterdata,compress_mask=compress_core.ShapeIntoTwoPower(np_filterdata,maskPre[icompressparam_index],pre_percentage,cur_percentage,max_compresstum_exp_,min_compresstum_exp_) 
            module.weight.data = torch.from_numpy(two_power_filterdata).cuda()
            #print(module.weight.data,compress_mask)  
            maskCur.append(torch.from_numpy(compress_mask).float().cuda())
            #two_power_filter_torch = torch.from_numpy(two_power_filterdata).cuda()
            and_maskCur = np.ones(compress_mask.shape)-compress_mask
            two_power_filter_torch_list.append(torch.from_numpy(and_maskCur).float().cuda()*module.weight.data)
            icompressparam_index +=1
            if module.bias is not None:
                torch_filterdata = module.bias.data
                #print(type(torch_filterdata),torch_filterdata.size(),torch_filterdata)
                np_filterdata = torch_filterdata.cpu().numpy()
                max_compresstum_exp_,min_compresstum_exp_ = compress_core.ComputeQuantumRange(np_filterdata,maskPre[icompressparam_index],7) 
                print('min and max:',max_compresstum_exp_,min_compresstum_exp_) 
                two_power_filterdata,compress_mask=compress_core.ShapeIntoTwoPower(np_filterdata,maskPre[icompressparam_index],pre_percentage,cur_percentage,max_compresstum_exp_,min_compresstum_exp_) 
                module.bias.data = torch.from_numpy(two_power_filterdata).cuda() 
                #print(module.bias.data,compress_mask)  
                maskCur.append(torch.from_numpy(compress_mask).float().cuda())
                and_maskCur = np.ones(compress_mask.shape)-compress_mask
                two_power_filter_torch_list.append(torch.from_numpy(and_maskCur).float().cuda()*module.bias.data)
                icompressparam_index +=1
    return maskCur,two_power_filter_torch_list
	
#compress train and val 
def compress_train_val(train_loader, val_loader,model, criterion, optimizer,args):
    global best_acc1
    maskCur,two_power_filter_torch_list=compress(model,args) #init generate mask
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, maskCur,two_power_filter_torch_list,args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'mask':maskCur
            }, is_best,args.snapshotmodelname)
def train(train_loader, model, criterion, optimizer, epoch,maskCur,two_power_filter_torch_list,args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        #update weight pow-of-two param do not change
        iconvparam_index = 0
        for module in model.modules():
            if isinstance(module,nn.Conv2d) or isinstance(module,nn.Linear):
				#two power param dont change 
                module.weight.data = module.weight.data * maskCur[iconvparam_index]+two_power_filter_torch_list[iconvparam_index]
                iconvparam_index+=1
                if module.bias is not None:
                    module.bias.data = module.bias.data * maskCur[iconvparam_index]+two_power_filter_torch_list[iconvparam_index]
                    iconvparam_index+=1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        break ###end simple test 


def validate(val_loader, model, criterion, args):
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

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
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
			
            break # for simple test

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename):
	fullpathname = filename+'_checkpoint.pth.tar'
	bestfilename = filename+'_model_best.pth.tar'
	torch.save(state, fullpathname)
	if is_best:
		shutil.copyfile(fullpathname, bestfilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
