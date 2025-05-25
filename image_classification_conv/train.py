from __future__ import print_function
import sys
import os

import argparse
import warnings
import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
from fvcore.common.timer import Timer

from utils import *

warnings.simplefilter(action='ignore', category=UserWarning)

def train(args, model, device, train_loader, optimizer, epoch, logger, start_index):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader, start_index):
        data, target = todevice(data, device), todevice(target, device)
        if args.optimizer in ["adam",'sgd','adamw']:

            optimizer.zero_grad()
            output = model(data)

            if args.loss == "cross_entropy":
                losses = [F.cross_entropy(output, target)]
            elif args.loss == "mse":
                losses = [F.mse_loss(output, target)]
            else:
                raise NotImplementedError
            
            loss = 0
            for l in losses:
                loss = loss + l
            loss.backward()
            optimizer.step()



        if batch_idx % args.log_interval == 0:

            with torch.no_grad():
                output = model(data)
                if args.loss == "cross_entropy":
                    losses = [F.cross_entropy(output, target)]
                elif args.loss == "mse":
                    losses = [F.mse_loss(output, target)]
                else:
                    raise NotImplementedError

                logger_info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: '.format(
                    epoch, (batch_idx - start_index) * len(data), len(train_loader.dataset),
                    100. * (batch_idx - start_index) / len(train_loader)) + ",".join([str(l.item()) for l in losses])
                logger.info(logger_info)
                
        if args.save_model_along and (batch_idx + 1) % args.save_model_interval == 0:
            torch.save(model.state_dict(), f"{args.exp_id}/{args.operation}_{batch_idx + 1}.pt")
            logger.info(f"model was saved to {args.exp_id}/{args.operation}_{batch_idx + 1}.pt")

        if args.dry_run:
            break

    return model

def test(args, model, device, test_loader, logger, name):
    model.eval()

    if args.loss == "cross_entropy":
        
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
    
    elif args.loss == "mse":
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = todevice(data, device), todevice(target, device)
                output = model(data)
                per_sample_loss = F.mse_loss(output, target, reduction='none')
                per_sample_rmse = torch.sqrt(per_sample_loss)
                test_loss += per_sample_rmse.sum().item()  # sum up batch loss

        test_loss /= len(test_loader.dataset)

        logger.info("\t"+name+' set: Average loss: {:.6f}'.format(test_loss))

        return test_loss
    
    else:
        raise NotImplementedError
    
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


    parser.add_argument('--dataset', type=str, default="Cifar10", #required=True,
                        help='dataset')    
    


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


    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1314,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model-interval', type = int, default=-1, 
                        help='whether save model along training')

    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--dropout_linear', type=float, default=0.5)
    parser.add_argument('--degree_out', type=int, default=1)
    parser.add_argument('--spline_order', type=int, default=3, 
                        help='order of the spline')
    parser.add_argument('--l1_penalty', type=float, default=0.0, 
                        help='L1 regularization penalty')
    parser.add_argument('--affine', action='store_true', default=True, 
                        help='whether to use affine transformation in normalization layers')
    parser.add_argument('--norm_layer', type=str, default='instance', 
                        help='type of normalization layer (batch, layer, instance)')
    ################# parameter for FastKAN #################
    parser.add_argument('--grid_size', type=int, default=8)
    ################# parameter for FastKAN #################
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    randomness_control(args.seed)
    args.device = device
    args.save_model_along = args.save_model_interval > 0

    if args.model in ['KAN','WavKAN','FastKAN']:
        args.layer_sizes = [8 * 4, 16 * 4, 32 * 4, 64 * 4]
    elif args.model in ['KAN8', 'WavKAN8', 'FastKAN8']:
        args.layers_sizes = [8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4]

    args.exp_id = f"../logs/{args.dataset}/{args.model}/"
    args.exp_id = args.exp_id + f"__{args.batch_size}__{args.epochs}__{args.lr}__{args.seed}__{args.groups}__{args.norm_layer}__{args.dropout}__{args.dropout_linear}__{args.degree_out}__{args.l1_penalty}__{args.affine}__{args.activation_name}__{args.optimizer}__{args.scheduler}__{args.gamma}"

    ################# id for KAN #################
    if args.model in ["KAN","KAN8"]:
        args.exp_id = args.exp_id + f"/{args.spline_order}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for KAN #################
    ################# id for FastKAN #################
    elif args.model in ["FastKAN","FastKAN8"]:
        args.exp_id = args.exp_id + f"/{args.grid_size}"
        os.makedirs(args.exp_id, exist_ok = True)
    ################# id for FastKAN #################

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

    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer =='adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)    

    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    elif args.scheduler =='cos':
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                            T_max =  -1) #  * iters         
        
    best_test_metric = 0 
    corresponding_train_metric = 0

    fvctimer = Timer()
    for epoch in range(1, args.epochs + 1):
        if fvctimer.is_paused():
            fvctimer.resume()
        else:
            fvctimer.reset()
        
        train(args, model, device, train_loader, optimizer, epoch, logger = logger, start_index = (epoch - 1) *len(train_loader))
        fvctimer.pause()
        train_metric = test(args, model, device, train_loader, logger = logger, name = "train")
        test_metric = test(args, model, device, test_loader, logger = logger, name = "test")
        

        if test_metric > best_test_metric:
            best_test_metric = test_metric
            corresponding_train_metric = train_metric


        scheduler.step()

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


    logger.info(f"training process was finished")

if __name__ == '__main__':
    main()
