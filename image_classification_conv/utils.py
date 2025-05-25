import logging, os, sys, gc, time, re
from datetime import datetime
import torch, random, numpy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, Subset
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy
import torchvision

from fvcore.nn import FlopCountAnalysis, parameter_count

from models.conv_kan_baseline import SimpleConvKAN,EightSimpleConvKAN
from models.fast_conv_kan_baseline import SimpleFastConvKAN, EightSimpleFastConvKAN
from models.metaconvkan import SimpleMetaConvKAN, EightSimpleMetaConvKAN
from models.metafastconvkan import SimpleMetaFastConvKAN, EightFastMetaConvKAN



def get_timestamp():
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    print('Log directory: ', log_dir)
    return logger, formatter

def get_loader(args, shuffle = True, use_cuda = True):

    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 0}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 0}

    if shuffle:
        train_kwargs.update({'shuffle': True})
        test_kwargs.update({'shuffle': False})
    else:
        train_kwargs.update({'shuffle': False})
        test_kwargs.update({'shuffle': False})

    if args.dataset == "MNIST":
        transform_train = v2.Compose([
            v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])

        transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
        # Load and transform the MNIST validation dataset
        test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_channel = 1 


    elif args.dataset == "Cifar10":
        transform_train = v2.Compose([

            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                             v2.AutoAugment(AutoAugmentPolicy.IMAGENET),
                             v2.AutoAugment(AutoAugmentPolicy.SVHN),
                             v2.TrivialAugmentWide()]),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        # Load and transform the CIFAR10 validation dataset
        test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        # Create DataLoaders for training and validation datasets


        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_channel = 3 
    elif args.dataset == "Cifar100":
        transform_train = v2.Compose([

            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                             v2.AutoAugment(AutoAugmentPolicy.IMAGENET),
                             v2.AutoAugment(AutoAugmentPolicy.SVHN),
                             v2.TrivialAugmentWide()]),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        # Load and transform the CIFAR100 validation dataset
        test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
        # Create DataLoaders for training and validation datasets
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 100
        input_channel = 3 


    else:
        raise NotImplementedError

    return train_loader, test_loader, num_classes, input_channel




def get_model(args):
    if args.model == "KAN":
        model = SimpleConvKAN(args)
    elif args.model == "FastKAN":
        model = SimpleFastConvKAN(args)
    elif args.model == "KAN8":
        model = EightSimpleConvKAN(args)
    elif args.model == "FastKAN8":
        model = EightSimpleFastConvKAN(args)
    elif args.model == "MetaKAN":
        model = SimpleMetaConvKAN(args)
    elif args.model == "MetaKAN8":
        model = EightSimpleMetaConvKAN(args)
    elif args.model == "MetaFastKAN":
        model = SimpleMetaFastConvKAN(args)
    elif args.model == "MetaFastKAN8":
        model = EightFastMetaConvKAN(args)
      
    else:
        raise NotImplementedError
    return model

def randomness_control(seed):
    print("seed",seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_matrix(matrix, path):
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='inferno')
    fig.colorbar(cax)
    fig.savefig(path)

def get_filename(path):
    base_name = os.path.basename(path)  # filename.extension
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension

def measure_time_memory(f):
    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_max_memory_allocated()
        else:
            start_memory = 0

        start_time = time.time()

        result = f(*args, **kwargs)

        end_time = time.time()

        if torch.cuda.is_available():
            end_memory = torch.cuda.max_memory_allocated()
        else:
            end_memory = 0

        print(f"Function {f.__name__} executed in {end_time - start_time:.4f} seconds.")
        print(f"Memory usage increased by {(end_memory - start_memory) / (1024 ** 2):.2f} MB to {(end_memory) / (1024 ** 2):.2f} MB.")
        
        return result
    return wrapped

def classwise_validation(logits, label, targets, args):
    accuracies = []
    for target in targets:
        accuracies.append(accuracy = get_accuracy(logits, label, target))
    return accuracies

def get_accuracy(probability, label, target = None):
    prediction = probability.max(dim = 1)[1]
    if target is None:
        return ((prediction == label).sum() / label.numel()).item()
    else:
        mask = label == target
        return ((prediction[mask]== label[mask]).sum() / label[mask].numel()).item()



def get_shortcut_function(args):
    if args.kan_shortcut_name == 'silu':
        return nn.SiLU()
    elif args.kan_shortcut_name == 'identity':
        return nn.Identity()
    elif args.kan_shortcut_name == 'zero':

        class Zero(nn.Module):
            def __init__(self):
                super(Zero, self).__init__()
            def forward(self, x):
                return x * 0

        return Zero()
    else:
        raise ValueError(f'Unknown kan shortcut function: {args.kan_shortcut_name}')
    
def get_model_complexity(model, logger, args, method = "fvcore"):

    parameter_dict = parameter_count(model)
    num_parameters = parameter_dict[""]


    if logger is not None:
        logger.info(f"Number of parameters: {num_parameters:,}")

    return num_parameters

def write_results(args, subfix = "", **kwargs):
    result_base = "./results"
    result_file = f"results{subfix}.csv"

    dataset, model, general_parameters, specific_parameter = args.exp_id.split("/")[2:]
    general_parameters = general_parameters.split("__")
    specific_parameter = specific_parameter.split("__")

    result_file_path = os.path.join(result_base, result_file)
    
    s = [get_timestamp(), dataset, model] + general_parameters + specific_parameter + [str(kwargs[key]) for key in kwargs]
    s = ",".join(s) + "\n"
    if not os.path.exists(os.path.dirname(result_file_path)):
        os.makedirs(os.path.dirname(result_file_path))
    with open(result_file_path, "a") as f:
        f.write(s)

def todevice(obj, device):
    if isinstance(obj, (list,tuple)):
        obj = [o.to(device) for o in obj]
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        raise NotImplementedError
    return obj