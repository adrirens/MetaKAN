from __future__ import print_function
import sys
import os

import argparse
import warnings
import torch
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
from fvcore.common.timer import Timer

from utils import *

warnings.simplefilter(action='ignore', category=UserWarning)

def train_double(args, model, device, train_loader, optimH, optimE, epoch, logger, start_index):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, start_index):
        data, target = todevice(data, device), todevice(target, device)

        optimH.zero_grad()
        optimE.zero_grad()
        output = model(data)


        losses = [F.cross_entropy(output, target)]

        
        loss = 0
        for l in losses:
            loss = loss + l
        loss.backward()
        optimE.step()
        optimH.step()

        if batch_idx % args.log_interval == 0:

            with torch.no_grad():
                output = model(data)

                losses = [F.cross_entropy(output, target)]


                logger_info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: '.format(
                    epoch, (batch_idx - start_index) * len(data), len(train_loader.dataset),
                    100. * (batch_idx - start_index) / len(train_loader)) + ",".join([str(l.item()) for l in losses])
                logger.info(logger_info)
                


    return model

def train_single(args, model, device, train_loader, optimizer, epoch, writer, logger, start_index):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, start_index):
        data, target = todevice(data, device), todevice(target, device)


        optimizer.zero_grad()
        output = model(data)

        losses = [F.cross_entropy(output, target)]

        loss = 0
        for l in losses:
            loss = loss + l
        loss.backward()
        optimizer.step()


        if batch_idx % args.log_interval == 0:

            with torch.no_grad():
                output = model(data)
                losses = [F.cross_entropy(output, target)]
                logger_info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: '.format(
                    epoch, (batch_idx - start_index) * len(data), len(train_loader.dataset),
                    100. * (batch_idx - start_index) / len(train_loader)) + ",".join([str(l.item()) for l in losses])
                logger.info(logger_info)

    return model

def test(model, device, test_loader,  logger, name):
    model.eval()
        
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = todevice(data, device), todevice(target, device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)   

    logger.info("\t"+name+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)
    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--model', type=str, default="KAN", #required=True,
                        help='network structure')

    parser.add_argument('--batch_norm', action='store_true', default=False,
                        help='whether use batch normalization')
    parser.add_argument('--activation_name', type=str, default="gelu", 
                        help='activation function')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='supported optimizer: adam, lbfgs')
    parser.add_argument('--scheduler', type=str, default="exponential",
                    help='scheduler of optimizer: adam, lbfgs')

    parser.add_argument('--optim_set', type=str, default="single",
                        help='optimizer settings for hypernet, option: single  double')

    parser.add_argument('--dataset', type=str, default="Cifar10", #required=True,
                        help='dataset')    
    
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id to use (default: 0)')    

    parser.add_argument('--batch-size', type=int, default=1024,
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, # 100 MNIST pretrain, 5 Finetune
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7, 1.0 for fewshot)')
    parser.add_argument('--loss', type=str, default="cross_entropy",
                        help='loss function')
    parser.add_argument('--wd_e', type=float, default=1e-4,
                        help='weight decay of embedding optimizer ')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1314,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')


    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--dropout_linear', type=float, default=0.5)
    parser.add_argument('--degree_out', type=int, default=1)

    parser.add_argument('--l1_penalty', type=float, default=0.0, 
                        help='L1 regularization penalty')
    parser.add_argument('--affine', action='store_true', default=True, 
                        help='whether to use affine transformation in normalization layers')
    parser.add_argument('--norm_layer', type=str, default='instance', 
                        help='type of normalization layer (batch, layer, instance)')


    ################# Parameters for hypernetwork #################
    parser.add_argument('--embedding_dim', type=int, default=3, 
                        help='dimension of embedding vector')  
    parser.add_argument('--hidden_dim',type=int, default= 32,
                        help='hidden dimension of hypernet')  
    parser.add_argument('--lr_h',type=float, default= 1e-3,
                        help='learning rate of hypernet')
    parser.add_argument('--lr_e',type=float, default= 1e-4,
                        help='learing rate of embedding')        
    ################# Parameters for hypernetwork #################
    ################# Parameters for MetaKAN #################
    parser.add_argument('--spline_order', type=int, default=3, 
                        help='order of the spline')
    
    ################# Parameters for MetaKAN #################
    ################# parameter for MetaFastKAN #################
    parser.add_argument('--grid_size', type=int, default=5)
    ################# parameter for MetaFastKAN #################    
    ################# Parameters for MetaWavKAN #################
    parser.add_argument('--wavelet_type', type=str, default='mexican_hat', 
                        help='mother wavlet funtion')  
    ################# Parameters for MetaWavKAN #################    

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")


    args.device = device

    randomness_control(args.seed)


    if args.model in ['MetaKAN','MetaWavKAN','MetaFastKAN']:
        args.layer_sizes = [8 * 4, 16 * 4, 32 * 4, 64 * 4]
    elif args.model in ['MetaKAN8', 'MetaWavKAN8', 'MetaFastKAN8']:
        args.layers_sizes = [8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4]

    args.exp_id = f"./logs/{args.dataset}/{args.model}/"

    args.exp_id = args.exp_id + f"{args.embedding_dim}__{args.hidden_dim}__{args.batch_size}__{args.epochs}__{args.seed}__{args.groups}__{args.norm_layer}__{args.dropout}__{args.dropout_linear}__{args.degree_out}__{args.l1_penalty}__{args.affine}__{args.activation_name}__{args.gamma}__{args.optimizer}__{args.scheduler}"
    if args.optim_set == 'double':
        args.exp_id = args.exp_id + f"{args.optim_set}__{args.lr_h}__{args.lr_e}__{args.wd_e}"
    elif args.optim_set == 'single':
        args.exp_id = args.exp_id + f"{args.optim_set}__{args.lr}"


    ################# id for MetaKAN #################
    if args.model in ["MetaKAN","MetaKAN8"]:
        args.exp_id = args.exp_id + f"/{args.spline_order}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for MetaKAN #################
    ################# id for MetaFastKAN #################
    elif args.model in ["MetaFastKAN","MetaFastKAN"]:
        args.exp_id = args.exp_id + f"/{args.grid_size}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for MetaFastKAN #################




    else:
        raise NotImplementedError
    
    if os.path.exists(os.path.join(args.exp_id, "log")):
        with open(os.path.join(args.exp_id, "log"), "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                if "training process was finished" in lines[-1]:
                    raise ValueError("training process was finished")

    logger, formatter = get_logger(args.exp_id, None, "log", level=logging.INFO)
    train_loader, test_loader, num_classes, input_channel = get_loader(args, use_cuda = use_cuda)

    args.num_classes = num_classes
    args.input_channel = input_channel



    model = get_model(args)

    logger.info(model)
    num_parameters = get_model_complexity(model, logger, args)
    model = model.to(device)
    if args.model =='MetaKAN':
        metanet = model.metanet
        embedding = model.embeddings
    elif args.model == 'MetaWavKAN':
        metanet = model.metanet
        embedding = model.embeddings
    elif args.model == 'MetaFastKAN':
        metanet = model.metanet
        embedding = model.embeddings

    if args.optim_set == 'double':

        optimH = optim.AdamW(metanet.parameters(), lr=args.lr_h, weight_decay=1e-4)
        optimE = optim.AdamW(embedding.parameters(), lr=args.lr_e, weight_decay=5e-4)
        if args.scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimH, gamma=args.gamma)
            scheduler_e =  optim.lr_scheduler.ExponentialLR(optimizer = optimE,
                                                                gamma=args.gamma) 
        elif args.scheduler =='cos':
            scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimH,
                                                                T_max =  -1) #  * iters 
            scheduler_e =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimE,

                                                                T_max =  -1) #  * iters 
    elif args.optim_set =='single':
        if args.optimizer == "adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
        elif args.optimizer =='adamw':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        else:
            raise NotImplementedError
        if args.scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

        elif args.scheduler =='cos':
            scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                                T_max =  -1) #  * iters 

    if args.loss == "cross_entropy":
        best_test_metric = 0 
    elif args.loss == "mse":
        best_test_metric = 1e10 
    else:
        raise NotImplementedError
    corresponding_train_metric = 0

    fvctimer = Timer()
    for epoch in range(1, args.epochs + 1):
        if fvctimer.is_paused():
            fvctimer.resume()
        else:
            fvctimer.reset()
        if args.optim_set == 'double':
            train_double(args, model, device, train_loader, optimH, optimE, epoch,  logger = logger, start_index = (epoch - 1) *len(train_loader))
        elif args.optim_set =='single':
            train_single(args, model, device, train_loader, optimizer, epoch, logger = logger, start_index = (epoch - 1) *len(train_loader))
        fvctimer.pause()
        train_metric = test(args, model, device, train_loader, logger = logger, name = "train", epoch= epoch)
        test_metric = test(args, model, device, test_loader,  logger = logger, name = "test", epoch= epoch)
        
        if args.loss == "cross_entropy":
            if test_metric > best_test_metric:
                best_test_metric = test_metric
                corresponding_train_metric = train_metric
        elif args.loss == "mse":
            if test_metric < best_test_metric:
                best_test_metric = test_metric
                corresponding_train_metric = train_metric
        else:
            raise NotImplementedError


        scheduler.step()
        if args.emb_sched:
            scheduler_e.step()

    total_training_time = fvctimer.seconds()
    average_training_time_per_epoch = fvctimer.avg_seconds()
    logger.info(f"total training time: {total_training_time:,} seconds; average training time per epoch: {average_training_time_per_epoch:,} seconds")

    write_results(
        args,
        train_metric = corresponding_train_metric,
        test_metric = best_test_metric,
        num_parameters = num_parameters,
        total_training_time = total_training_time,
        average_training_time_per_epoch = average_training_time_per_epoch
    )

    if args.save_model:
        torch.save(
            {   
                "args" : args,
                "state_dict" : model.state_dict(),
                "metrics" : {
                    "train_metric" : corresponding_train_metric,
                    "test_metric" : best_test_metric,
                    "num_parameters" : num_parameters,
                    "total_training_time" : total_training_time,
                    "average_training_time_per_epoch" : average_training_time_per_epoch
                }
            }, f"{args.exp_id}/ckpt.pt")
        logger.info(f"model was saved to {args.exp_id}/ckpt.pt")

    logger.info(f"training process was finished")

if __name__ == '__main__':
    main()