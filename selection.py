import argparse
import os,sys
import numpy as np
import random 
import  csv, ast
import  matplotlib.pyplot as plt
from PIL import Image
import copy
from tqdm import tqdm
import sklearn
import time

import torch 
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torchvision.models as models
import mymodels 
from scipy.spatial import distance

import torch_geometric
from torch_geometric.nn import knn_graph
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
from torch_cluster import knn
from typing import Optional

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from scipy.special import softmax
from scipy.spatial.distance import pdist, squareform
import pandas as pd

# import networkx as nx 
from gragh import propogate
from utils import AverageMeter, RecorderMeter

import data 

# device: gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# CUDA_LAUNCH_BLOCKING=1


model_names = sorted(name for name in mymodels.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(mymodels.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser("Arguments for sample selection.")
parser.add_argument('--data_path',
                    default='../../../datasets/',
                    type=str,
                    help='Path to dataset')
parser.add_argument('--dataset',
                    type=str,
                    choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet-200', 'svhn', 'stl10', 'mnist', 'emnist', 'gtsrb'],
                    help='Choices: cifar10, cifar100, imagenet, svhn, stl10, mnist, gtsrb.')
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='Batch size')
parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='Batch size')
# test model
parser.add_argument('--model2test_arch',
                    metavar='ARCH',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')

parser.add_argument("--model2test_path", 
                        type=str, 
                        default=None, 
                        help=" model of test center (resnet18, resnet50)")
parser.add_argument("--model_number", type=int, 
                         help='choose from three different settings: all ones, random 1, random 2')
   

# test config
# baseline
parser.add_argument("--baseline_gini",
                    action='store_true',
                    help='using gini technique to select test cases')
parser.add_argument("--baseline_mcp",
                    action='store_true',
                    help='using mcp technique to select test cases')
parser.add_argument("--baseline_dsa",
                    action='store_true',
                    help='using distance surprise adaquacy technique to select test cases')
parser.add_argument("--baseline_uncertainty",
                    action='store_true',
                    help='using dropout technique to estimate the uncertainty for selecting test cases')
parser.add_argument("--baseline_random",
                    action='store_true',
                    help='randomly select test cases, only implemented for retraining')

parser.add_argument("--sel_method", 
                        type=str, 
                        default=None, 
                        help="[kmeans, random]")
parser.add_argument("--feature_extractor_id",
                        type=int, default=1, help="available choices: [0. unsupervised BYOL model, 1. model under test")
parser.add_argument("--graph_nn", 
                        action="store_true",
                        help="use graph network to generate test selection")
parser.add_argument("--no_neighbors",
                        type=int,
                        default=100,
                        help='number of neighbors to construct the knn graph')
# direct and indirect feedback combine method
parser.add_argument("--bf_mixed",
                    action="store_true",
                    help="combine direct and indirect correlation with a brute force way (multiplication)")
parser.add_argument("--learn_mixed",
                    action="store_true",
                    help='learn to combine direct and indirect correlation')

parser.add_argument("--latent_space_plot", 
                        action='store_true',  
                        help='plot all latent space or not.')
parser.add_argument("--supervised_latents", 
                        action='store_true',
                        help='supervised or unsupervised encoder')
parser.add_argument("--save_path", 
                        type=str, 
                        default='', help='the file to store the hit ratios')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
##########################################################################


args = parser.parse_args()

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if device == 'cuda':
    torch.cuda.manual_seed(args.manualSeed)


default_hyperparams = {
    'custom':  {'img_size':32,'num_classes':10, 'channel':3, 'gnn_hidden_dim': 32, 'gnn_train_epoch':600, 'retrain_epoch':10, 'feature_dim':512},
    'svhn':    {'img_size':32,'num_classes':10, 'channel':3, 'gnn_hidden_dim': 32, 'gnn_train_epoch':600, 'retrain_epoch':10, 'feature_dim':512},
    'cifar10': {'img_size':32,'num_classes':10, 'channel':3, 'gnn_hidden_dim': 32, 'gnn_train_epoch':600, 'retrain_epoch':10, 'feature_dim':512},
    'stl10':   {'img_size':96,'num_classes':10, 'channel':3, 'gnn_hidden_dim': 32, 'gnn_train_epoch':600, 'retrain_epoch':10, 'feature_dim':512},
    
}

weight_per_class_1ist = [np.ones(10), 
                        np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701,
                            0.22479665, 0.19806286, 0.76053071, 0.16911084, 0.08833981]), 
                        np.array([0.68535982, 0.95339335, 0.00394827, 0.51219226, 0.81262096,
                            0.61252607, 0.72175532, 0.29187607, 0.91777412, 0.71457578])]

weight_per_class  = weight_per_class_1ist[args.model_number]

retrain_epoch = default_hyperparams[args.dataset]['retrain_epoch']
no_neighbors = args.no_neighbors 

###############################################################################
###############################################################################
def main():
    global retrain_epoch
    global no_neighbors
    

    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)
    
    print_log(args, log)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        num_classes = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        num_classes = 100
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        num_classes = 10
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'stl10':
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    
    data_dir = os.path.join(args.data_path, args.dataset)
    img_size = default_hyperparams[args.dataset]['img_size']

    if args.dataset == 'svhn':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
        train_set = torchvision.datasets.SVHN(root=data_dir, 
                                                split = 'train',
                                                transform=train_transform, 
                                                download=True)     
        num_train = len(train_set)
        train_idx = list(range(num_train))[:40000]
        val_idx = list(range(num_train))[40000:50000]
        sub_train_set = torch.utils.data.Subset(train_set, train_idx)   

    elif args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
            
        train_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                train=True, 
                                                transform=train_transform, 
                                                download=True)
        num_train = len(train_set)
        train_idx = list(range(num_train))[:15000]
        val_idx = list(range(num_train))[15000:20000]
        sub_train_set = torch.utils.data.Subset(train_set, train_idx)   

    elif args.dataset == 'stl10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
            
        train_set = torchvision.datasets.STL10(root=data_dir, 
                                                split='train', 
                                                transform=train_transform, 
                                                download=True)
        num_train = len(train_set)
        train_idx = list(range(num_train))[:4000]
        val_idx = list(range(num_train))[4000:5000]
        sub_train_set = torch.utils.data.Subset(train_set, train_idx)   

    else:
        raise ValueError('Invalid dataset')

    #####################################################
    # ---- Test Center: Receive models under test------
    print_log("=> creating model 2 test'{}'".format(args.model2test_arch), log)
    # Init model, criterion, and optimizer
    model2test = mymodels.__dict__[args.model2test_arch](num_classes=num_classes, channels=default_hyperparams[args.dataset]['channel']).to(device)
    print_log("=> network :\n {}".format(model2test), log)
    # load weights
    checkpoint = torch.load(args.model2test_path, map_location=device)
    model2test.load_state_dict(checkpoint['net'])
    
    # restore the trainset training the model
    dataset_save_path = os.path.join('./checkpoint', 
                                        args.dataset, 
                                        'ckpt_bias', 
                                        'biased_dataset',
                                        str(args.model_number))
    sub_train_index = np.load(os.path.join(dataset_save_path, 'train.npy'))
    sub_val_index = np.load(os.path.join(dataset_save_path, 'val.npy'))

    # sub_train_index = np.array(sub_train_index, dtype=np.int32)
    # sub_val_index = np.array(sub_val_index, dtype=np.int32)
    # print (sub_train_index, sub_val_index)
    model2test_trainset = torch.utils.data.Subset(sub_train_set, sub_train_index)

    #####################################################
    # ---- Test Center: Prepare Test Dataset------
    #  get testset and testloader 
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    if args.dataset == 'cifar10':
        # test data : total 30k + 10k = 40k
        train_set = torchvision.datasets.CIFAR10(
                                    root=data_dir,
                                    train=True,
                                    transform=test_transform,
                                    download=False )
        test_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                train=False,
                                                transform=test_transform,
                                                download=False)
        mix_test_set = torch.utils.data.ConcatDataset([train_set, test_set])
        indices = list(range(len(mix_test_set)))
        # train_indices = np.array(indices[:20000])
        T2_indices = np.array(indices[20000:21000]) # 1k images

        T2_set = torch.utils.data.Subset(mix_test_set, T2_indices)
        # labeled case number
        rest_indices = np.array(indices[21000:])
        test_set = torch.utils.data.Subset(mix_test_set, rest_indices)

        #split test dataset into labeled and unlabeled parts
        num_test = len(test_set)
        indices = list(range(num_test))
        
        labeled_indices = np.array(indices[:7800]) # 8k image
        unlabeled_indices = np.array(indices[7800:]) 
        
        print_log("dataset segmentation: labeled/unlabled/T2: {} {} {}".format(labeled_indices.size, unlabeled_indices.size, T2_indices.size), log)
        

    elif args.dataset == 'stl10':
        # test data :  
        train_set = torchvision.datasets.STL10(
                                    root=data_dir,
                                    split='train',
                                    transform=test_transform,
                                    download=False )
        test_set = torchvision.datasets.STL10(root=data_dir, 
                                                split='test',
                                                transform=test_transform,
                                                download=False)
        mix_test_set = torch.utils.data.ConcatDataset([train_set, test_set])
        indices = list(range(len(mix_test_set)))
        T2_indices = np.array(indices[5000:5500]) # 1k images

        rest_indices = np.array(indices[5500:])

        T2_set = torch.utils.data.Subset(mix_test_set, T2_indices)
        test_set = torch.utils.data.Subset(mix_test_set, rest_indices)

        # split test dataset into labeled and unlabeled parts
        num_test = len(test_set)
        indices = list(range(num_test))
        
        labeled_indices = np.array(indices[:int(0.2*num_test)]) # 8k images
        unlabeled_indices = np.array(indices[int(0.2*num_test):]) # 1w images
        print_log("dataset segmentation: labeled/unlabled/T2: {} {} {}".format(labeled_indices.size, unlabeled_indices.size, T2_indices.size), log)

    elif args.dataset == 'svhn': 
        # test data: unlabeled 3w, labeled 1w, hold-out test data 53w
        train_set = torchvision.datasets.SVHN(root=data_dir,
                                                split='train',
                                                transform=test_transform,
                                                download=True)
        test_set = torchvision.datasets.SVHN(root=data_dir,
                                                split='test',
                                                transform=test_transform,
                                                download=True)
        T2_set = torchvision.datasets.SVHN(root=data_dir,
                                            split='extra',
                                            transform=test_transform,
                                            download=True)
        mix_test_set = torch.utils.data.ConcatDataset([train_set, test_set])

        num_test = len(mix_test_set)
        indices = list(range(num_test))
        rest_indices = np.array(indices[50000:])

        test_set = torch.utils.data.Subset(mix_test_set, rest_indices)
        labeled_indices = np.arange(len(test_set))[:10000]
        unlabeled_indices = np.arange(len(test_set))[10000:]
        print_log("dataset segmentation: labeled/unlabled/T2: {} {} {}".format(labeled_indices.size, unlabeled_indices.size, len(T2_set)), log)

    else:
        print_log("not supported dataset", log)
        exit(0)

    
    ####################### A pre-evaluation ############
    # Accuracy before retrain
    # T2: test cases in deployment
    T2_acc = 0
    # if args.retrain:
    print_log ("# of hold-out test inputs: {}".format(len(T2_set)), log)
    T2_acc = test_acc(T2_set, model2test)
    print_log("ACC in Hold-Out dataset: {}".format(round(T2_acc, 2)), log)

    
    ####################### Test Selection ############
    # percentage of budget over all test cases
    p_budget_lst = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100] # percentage of budget
    budget_lst = [] 
    pfd_lst = []
    ideal_pfd_lst = []
    budget_lst.insert(0, 0)
    pfd_lst.insert(0, 0)
    ideal_pfd_lst.insert(0, 0)


    if args.baseline_random:
        unlabeled_test_inputs = torch.utils.data.Subset(test_set, unlabeled_indices)
        print_log ("# of unlabeled test inputs: {}".format(len(unlabeled_test_inputs)), log)

        correct_array, logits  = test(unlabeled_test_inputs, model2test, 
                                num_classes=num_classes)
        misclass_array = (correct_array==0).astype(int)

        ranked_indexes = list(np.arange(len(unlabeled_test_inputs)))
        random.shuffle(ranked_indexes)

        # test
        for p_budget in p_budget_lst:
            print_log("\n ###### budget percent is {} ######".format(p_budget), log)
            budget = int(p_budget*len(unlabeled_indices)/100.0)
            model2test_temp = copy.deepcopy(model2test)

            selected = set(ranked_indexes[:budget])

            # write results and logging
            out_file = os.path.join(args.save_path, 'random_result.csv')
            print_log ("writing output to csv file: {}".format(out_file), log)

            with open(out_file, 'a+') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([args.model2test_arch, 
                                args.model2test_path,
                                budget, 
                                p_budget,
                                args.sel_method])

            print_log('success!', log)
        log.close()

        return
    # baseline 1
    if args.baseline_gini:
        # DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks
        unlabeled_test_inputs = torch.utils.data.Subset(test_set, unlabeled_indices)
        print_log ("# of unlabeled test inputs: {}".format(len(unlabeled_test_inputs)), log)

        correct_array, logits  = test(unlabeled_test_inputs, model2test, 
                                num_classes=num_classes)
        # get classification result rank
        prob = softmax(logits, axis=1)
        pred = np.sum(np.square(prob), axis=1)

        ranked_indexes = np.argsort(pred)

        # gt
        misclass_array = (correct_array==0).astype(int)

        # plot histogram: relation between score and correctness 
        bins = np.linspace(0, 1, 100)
        plt.hist(pred, bins, alpha=0.3, label='total')
        plt.hist(pred[correct_array.astype(bool)], bins, log=True, alpha=0.5, color='b', label='correct')
        plt.hist(pred[misclass_array.astype(bool)], bins, log=True, alpha=0.5, color='r', label='misclassified')
        plt.legend()
        plt.savefig(os.path.join(args.save_path, 'confidence_label.pdf'))

        # test
        for p_budget in p_budget_lst:
            print_log("\n ###### budget percent is {} ######".format(p_budget), log)
            budget = int(p_budget*len(unlabeled_indices)/100.0)
            model2test_temp = copy.deepcopy(model2test)

            selected = ranked_indexes[:budget]
            pos_count = misclass_array[selected].sum()
            # store result 
            p_fault_detected = 100.0*pos_count/misclass_array.sum()
            ideal_fault_detected = 100.0*budget/misclass_array.sum()
            if ideal_fault_detected > 100.0:
                ideal_fault_detected = 100.0
            random_p_fault_detected = (100.0*budget/misclass_array.shape[0])
   
            print_log("Percentage of fault detected: %s "%(p_fault_detected), log)
            print_log("Percentage of fault detected (random): %s "%(random_p_fault_detected), log)
  
            # write results and logging
            out_file = os.path.join(args.save_path, 'gini_result.csv')
            print_log ("writing output to csv file: {}".format(out_file), log)

            budget_lst.append(p_budget/100.0)
            pfd_lst.append(p_fault_detected/100.0)
            ideal_pfd_lst.append(ideal_fault_detected/100.0)
            
            apfd = 0
            ideal_apfd = 0
            if p_budget == 100:
                print (pfd_lst, ideal_pfd_lst)
                apfd = get_APFD(copy.copy(budget_lst), pfd_lst)
                ideal_apfd = get_APFD(budget_lst, ideal_pfd_lst)

            with open(out_file, 'a+') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([args.model2test_arch, 
                                args.model2test_path,
                                budget, 
                                p_budget,
                                args.sel_method, 
                                'FaultDetected',
                                p_fault_detected,
                                ideal_fault_detected,
                                random_p_fault_detected,
                                'APFD',
                                apfd,
                                ideal_apfd,
                                'TRC',
                                p_fault_detected/ideal_fault_detected])

            print_log('success!', log)
        log.close()

        return

    if args.baseline_mcp:
        import mcp
        # Multiple-Boundary Clustering and Prioritization to Promote Neural Network Retraining
        unlabeled_test_inputs = torch.utils.data.Subset(test_set, unlabeled_indices)
        print_log ("# of unlabeled test inputs: {}".format(len(unlabeled_test_inputs)), log)
        # gt
        correct_array, _ = test(unlabeled_test_inputs, model2test,
                                    num_classes=num_classes)
        misclass_array = (correct_array==0).astype(int)

        _, logits = test(unlabeled_test_inputs, model2test, num_classes=num_classes)
        prob = softmax(logits, axis=1)
        dicratio=[[] for i in range(num_classes*num_classes)]
        dicindex=[[] for i in range(num_classes*num_classes)]
        for i in range(len(prob)):
            act=prob[i]
            max_index,sec_index,ratio = mcp.get_boundary_priority(act)#max_index 
            dicratio[max_index*num_classes+sec_index].append(ratio)
            dicindex[max_index*num_classes+sec_index].append(i)
        
        for p_budget in p_budget_lst:
            print_log("\n ###### budget percent is {} ######".format(p_budget), log)
            model2test_temp = copy.deepcopy(model2test)
            dicindex_temp = copy.deepcopy(dicindex)
            dicratio_temp = copy.deepcopy(dicratio)

            budget = int(p_budget*len(unlabeled_indices)/100.0)
            if p_budget == 100:
                selected = list(np.arange(len(unlabeled_indices)))
            else:
                selected = mcp.select_from_firstsec_dic(budget, dicratio_temp, dicindex_temp, num_classes=num_classes)


            # test
            pos_count = misclass_array[selected].sum()
            # store result 
            p_fault_detected = 100.0*pos_count/misclass_array.sum()
            ideal_fault_detected = 100.0*budget/misclass_array.sum()
            if ideal_fault_detected > 100.0:
                ideal_fault_detected = 100.0
            random_p_fault_detected = (100.0*budget/misclass_array.shape[0])

            print_log("Percentage of fault detected: %s "%(p_fault_detected), log)
            print_log("Percentage of fault detected (random): %s "%(random_p_fault_detected), log)

            # write results and logging
            out_file = os.path.join(args.save_path, 'mcp_result.csv')
            print_log ("writing output to csv file: {}".format(out_file), log)

            budget_lst.append(p_budget/100.0)
            pfd_lst.append(p_fault_detected/100.0)
            ideal_pfd_lst.append(ideal_fault_detected/100.0)

            apfd = 0
            ideal_apfd = 0
            if p_budget == 100:
                # print (budget_lst, pfd_lst, ideal_pfd_lst)
                apfd = get_APFD(copy.copy(budget_lst), pfd_lst)
                ideal_apfd = get_APFD(budget_lst, ideal_pfd_lst)

            with open(out_file, 'a+') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([args.model2test_arch, 
                                args.model2test_path,
                                budget, 
                                p_budget,
                                args.sel_method, 
                                'FaultDetected',
                                p_fault_detected,
                                ideal_fault_detected,
                                random_p_fault_detected,
                                'APFD',
                                apfd,
                                ideal_apfd,
                                'TRC',
                                p_fault_detected/ideal_fault_detected])

            print_log('success!', log)
        log.close()

        return


    if args.baseline_dsa:
        from keras import backend as K
        from pytorch2keras import pytorch_to_keras
        import dsa
        from torch.autograd import Variable
        
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        K.set_image_data_format('channels_first')
        print_log('{} {}'.format(K.image_data_format(), K.backend()), log)
        # implementation of surprise adaquacy as selection method
        print_log("Select test case with surprise adaquacy", log)
        unlabeled_test_inputs = torch.utils.data.Subset(test_set, unlabeled_indices)
        print_log("# of unlabeled test inputs: {}".format(len(unlabeled_test_inputs)), log)
        # gt
        correct_array, _ = test(unlabeled_test_inputs, model2test,
                                    num_classes=num_classes)
        misclass_array = (correct_array==0).astype(int)

        # preprare keras model, dataset, for dsa calculation
        # dummy variable
        channel = default_hyperparams[args.dataset]['channel']
        image_size = default_hyperparams[args.dataset]['img_size']
        print (channel, image_size)
        input_np = np.random.uniform(0,1, (1, channel, image_size, image_size))
        input_var = Variable(torch.FloatTensor(input_np))
        pytorch_output = model2test(input_var.to(device))[0].cpu().detach().numpy()
        k_model = pytorch_to_keras(model2test, input_var.to(device), [(channel, image_size, image_size,)],
                                    verbose=True,
                                    name_policy='short')
        print_log(k_model.summary(), log)
        keras_output = k_model.predict(input_np)[0]
        # print (keras_output.shape, pytorch_output.shape)
        error = np.max(pytorch_output - keras_output)
        print_log('{} {} {}'.format(pytorch_output, '\n', keras_output), log)
        print ("error, ", error)

        # check again: to be deleted
        input_np = np.random.uniform(0, 1, (1, channel, image_size, image_size))
        input_var = Variable(torch.FloatTensor(input_np))
        p_output = model2test(input_var.to(device))[0].cpu().detach().numpy()
        k_output = k_model.predict(input_np)[0]
        print (p_output, k_output)
        print (np.sum(p_output - k_output))

        # sda ranking
        def dataset_convertion(input_data):
            output_data = [input_data[i][0].numpy() for i in range(len(input_data))]
            output_label = [input_data[i][1] for i in range(len(input_data))]
            # print (output_data)
            output_data = np.array(output_data)
            output_label = np.array(output_label)
            return output_data, output_label
            
        keras_model_train_data = dataset_convertion(model2test_trainset)[0]
        keras_unlabeled_data = dataset_convertion(unlabeled_test_inputs)[0]
        args.num_classes = num_classes

        if args.model2test_arch == 'resnet18':
            layer_names = ['189']
        else:
            layer_names = ['output_1']
        
        dsa_values = dsa.fetch_dsa(model=k_model, 
                                x_train=keras_model_train_data, 
                                x_target=keras_unlabeled_data, 
                                target_name='', layer_names=layer_names, args=args)

        # higher dsa values in the front
        ranked_indexes = np.argsort(dsa_values)[::-1]
        print (ranked_indexes)
        print ("dsa values: ", dsa_values[ranked_indexes[0]], dsa_values[ranked_indexes[1]], dsa_values[ranked_indexes[2]])
        # return 

        for p_budget in p_budget_lst:
            print_log("\n ###### budget percent is {} ######".format(p_budget), log)
            budget = int(p_budget*len(unlabeled_indices)/100.0)
            model2test_temp = copy.deepcopy(model2test)

            selected = ranked_indexes[:budget]
            pos_count = misclass_array[selected].sum()
            # store result 
            p_fault_detected = 100.0*pos_count/misclass_array.sum()
            ideal_fault_detected = 100.0*budget/misclass_array.sum()
            if ideal_fault_detected > 100.0:
                ideal_fault_detected = 100.0
            random_p_fault_detected = (100.0*budget/misclass_array.shape[0])

            print_log("Percentage of fault detected: %s "%(p_fault_detected), log)
            print_log("Percentage of fault detected (random): %s "%(random_p_fault_detected), log)

            # write results and logging
            out_file = os.path.join(args.save_path, 'dsa_result.csv')
            print_log ("writing output to csv file: {}".format(out_file), log)

            budget_lst.append(p_budget/100.0)
            pfd_lst.append(p_fault_detected/100.0)
            ideal_pfd_lst.append(ideal_fault_detected/100.0)
            
            apfd = 0
            ideal_apfd = 0
            if p_budget == 100:
                print (pfd_lst, ideal_pfd_lst)
                apfd = get_APFD(copy.copy(budget_lst), pfd_lst)
                ideal_apfd = get_APFD(budget_lst, ideal_pfd_lst)

            with open(out_file, 'a+') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([args.model2test_arch, 
                                args.model2test_path,
                                budget, 
                                p_budget,
                                args.sel_method, 
                                'FaultDetected',
                                p_fault_detected,
                                ideal_fault_detected,
                                random_p_fault_detected,
                                'APFD',
                                apfd,
                                ideal_apfd,
                                'TRC',
                                p_fault_detected/ideal_fault_detected])

            print_log('success!', log)
        log.close()
        
        return 
    
    if args.baseline_uncertainty:
        
        # implementation of dropout uncertainty
        print_log ("Select test case with dropout uncertainty", log)
        unlabeled_test_inputs = torch.utils.data.Subset(test_set, unlabeled_indices)
        print_log ("# of unlabeled test inputs: {}".format(len(unlabeled_test_inputs)), log)
        # gt
        correct_array, _ = test(unlabeled_test_inputs, model2test,
                                    num_classes=num_classes)
        misclass_array = (correct_array==0).astype(int)

        def uncertainty_test(model, test_set):
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=16)
            model.train()
            model.dropout = True
            T = 100
            unct_list_v1 = []
            unct_list_v2 = []
            for data, target in test_loader:
                output_list = []
                data, target = data.to(device), target.to(device)
                for i in range(T):
                    output = model(data)[0].detach()
                    output_list.append(torch.unsqueeze(F.softmax(output, dim=1), 0))
                # print (torch.cat(output_list, 0).var(0).mean(1).shape)
                output_mean = torch.cat(output_list, 0).mean(0).cpu().numpy()
                # v1: variance
                output_var_v1 = torch.cat(output_list, 0).var(0).mean(1).data.cpu().numpy()
                # v2: mean and entropy
                output_var_v2 = - np.sum(output_mean * np.log(output_mean), axis=1)
                unct_list_v1.extend(output_var_v1)
                unct_list_v2.extend(output_var_v2)
            model.dropout = False
            return unct_list_v1, unct_list_v2

        unct_list_v1, unct_list_v2 = uncertainty_test(model2test, unlabeled_test_inputs)
        print_log('uncertainty values: {}'.format(unct_list_v2), log)

        # use v2
        ranked_indexes = np.argsort(np.array(unct_list_v2))[::-1]
        print_log(np.array(unct_list_v2).shape, log)
        
        # plot histogram: relation between uncertainty and correctness 
        bins = np.linspace(min(unct_list_v2), max(unct_list_v2), 100)
        plt.hist(np.array(unct_list_v2), bins, alpha=0.3, label='total')
        plt.hist(np.array(unct_list_v2)[correct_array.astype(bool)], bins, log=True, alpha=0.5, color='b', label='correct')
        plt.hist(np.array(unct_list_v2)[misclass_array.astype(bool)], bins, log=True, alpha=0.5, color='r', label='misclassified')
        plt.legend()
        plt.savefig(os.path.join(args.save_path, 'uncertainty_label.pdf'))

        for p_budget in p_budget_lst:
                print_log("\n ###### budget percent is {} ######".format(p_budget), log)
                model2test_temp = copy.deepcopy(model2test)
                budget = int(p_budget*len(unlabeled_indices)/100.0)

                selected = ranked_indexes[:budget]
                pos_count = misclass_array[selected].sum()
                # store result 
                p_fault_detected = 100.0*pos_count/misclass_array.sum()
                ideal_fault_detected = 100.0*budget/misclass_array.sum()
                if ideal_fault_detected > 100.0:
                    ideal_fault_detected = 100.0
                random_p_fault_detected = (100.0*budget/misclass_array.shape[0])
                
                print_log("Percentage of fault detected: %s "%(p_fault_detected), log)
                print_log("Percentage of fault detected (random): %s "%(random_p_fault_detected), log)

                # write results and logging
                out_file = os.path.join(args.save_path, 'dropout_uncertainty_result.csv')
                print_log ("writing output to csv file: {}".format(out_file), log)

                budget_lst.append(p_budget/100.0)
                pfd_lst.append(p_fault_detected/100.0)
                ideal_pfd_lst.append(ideal_fault_detected/100.0)

                apfd = 0
                ideal_apfd = 0
                if p_budget == 100:
                    # print (budget_lst, pfd_lst, ideal_pfd_lst)
                    apfd = get_APFD(copy.copy(budget_lst), pfd_lst)
                    ideal_apfd = get_APFD(budget_lst, ideal_pfd_lst)

                with open(out_file, 'a+') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([args.model2test_arch, 
                                    args.model2test_path,
                                    budget, 
                                    p_budget,
                                    args.sel_method, 
                                    'FaultDetected',
                                    p_fault_detected,
                                    ideal_fault_detected,
                                    random_p_fault_detected,
                                    'APFD',
                                    apfd,
                                    ideal_apfd,
                                    'TRC',
                                    p_fault_detected/ideal_fault_detected
                                    ])

                print_log('success!', log)
        log.close()
            
        return 

    #####################################################
    # ------ Test Center: Get the result ground truth -----
    # Test model2test use all test cases
    print_log('Get the ground truth', log)
    correct_array, logits = test(test_set, model2test, num_classes=num_classes)
    misclass_array = (correct_array==0).astype(int)

    # ---- Test Center: Get latent vectors of test dataset------
    fe_config = ['unsup', 'self']
    feature_extractor = fe_config[args.feature_extractor_id]

    if feature_extractor == 'unsup':
        print_log('Use unsupervised BYOL model as feature extractor', log)
        if args.dataset == 'cifar10':
            unsup_img_size = 224
            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=unsup_img_size),
                torchvision.transforms.ToTensor()])
            train_set_t = torchvision.datasets.CIFAR10(
                                        root=data_dir,
                                        train=True,
                                        transform=test_transform,
                                        download=False )
            test_set_t = torchvision.datasets.CIFAR10(root=data_dir, 
                                                    train=False,
                                                    transform=test_transform,
                                                    download=False)
            mix_test_set_t = torch.utils.data.ConcatDataset([train_set_t, test_set_t])
            test_set_t = torch.utils.data.Subset(mix_test_set_t, rest_indices)  
            feature_extractor_net = models.resnet18(pretrained=False).to(device)
            feature_extractor_net.load_state_dict(torch.load('./byol/cifar10_fc_224.pt', map_location=device))
        if args.dataset == 'stl10':
            unsup_img_size = 96 
            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=unsup_img_size),
                torchvision.transforms.ToTensor()])
            train_set_t = torchvision.datasets.STL10(
                                        root=data_dir,
                                        split='train',
                                        transform=test_transform,
                                        download=False )
            test_set_t = torchvision.datasets.STL10(root=data_dir, 
                                                    split='test',
                                                    transform=test_transform,
                                                    download=False)
            mix_test_set_t = torch.utils.data.ConcatDataset([train_set_t, test_set_t])
            test_set_t = torch.utils.data.Subset(mix_test_set_t, rest_indices)  
            feature_extractor_net = models.resnet18(pretrained=False, num_classes=num_classes).to(device)
            feature_extractor_net.load_state_dict(torch.load('./byol/stl10_fe_96.pt', map_location=device))
        elif args.dataset == 'svhn':
            unsup_img_size = 224
            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=unsup_img_size),
                torchvision.transforms.ToTensor()])
            train_set_t = torchvision.datasets.SVHN(root=data_dir,
                                                    split='train',
                                                    transform=test_transform,
                                                    download=False)
            test_set_t = torchvision.datasets.SVHN(root=data_dir,
                                                  split='test', 
                                                  transform=test_transform,
                                                  download=False)
            mix_test_set_t = torch.utils.data.ConcatDataset([train_set_t, test_set_t])
            test_set_t = torch.utils.data.Subset(mix_test_set_t, rest_indices)
            feature_extractor_net = models.resnet18(pretrained=False, num_classes=num_classes).to(device)
            feature_extractor_net.load_state_dict(torch.load('./byol/svhn_fe_224.pt', map_location=device))
        elif args.dataset == 'emnist':
            unsup_img_size = 32
            test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=unsup_img_size),
                torchvision.transforms.ToTensor()])
            train_set_t = torchvision.datasets.EMNIST(root=data_dir,
                                                    split='bymerge',
                                                    train=True,
                                                    transform=test_transform,
                                                    download=False)
            test_set_t = torchvision.datasets.EMNIST(root=data_dir,
                                                  split='bymerge', 
                                                  train=False,
                                                  transform=test_transform,
                                                  download=False)
            mix_test_set_t = torch.utils.data.ConcatDataset([train_set_t, test_set_t])
            test_set_t = torch.utils.data.Subset(mix_test_set_t, rest_indices)
            feature_extractor_net = mymodels.small_resnet10(pretrained=False, channels=1).to(device)
            feature_extractor_net.load_state_dict(torch.load('./byol/checkpoints/official-{}/resnet18-model-95-{}.pt'.format(args.dataset, unsup_img_size), map_location=device))


        # remove the fc layer
        feature_extractor_net = nn.Sequential(*list(feature_extractor_net.children())[:-1])
        latents = get_features(test_set_t, feature_extractor_net)

    elif feature_extractor == 'self':
        # bug 512*8*8 instead of 512?
        print_log("Use model2test as feature extractor", log)
        feature_extractor_net = model2test
        feature_extractor_net = nn.Sequential(*list(feature_extractor_net.children())[:-1])
        latents = get_features(test_set, feature_extractor_net)
        latents = latents.reshape(len(test_set), -1)
    else:
        raise ValueError("Unknown feature extractor configuration")

    #latents = sklearn.preprocessing.normalize(latents, norm='l2')
    # print_log("latents shape {}".format(latents.shape), log)


    #####################################################
    # # ---- Test Center: Visulize test latent vectors------
    # tSNE
    full_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    labels = next(iter(full_loader))[1]
    print ("labels: ", labels[:100])
    class_inds = labels >= 8
    print ("boolean class indexes: ", class_inds)    
    print ('length of selected visualize instances: ', class_inds.sum())  
    
    if args.latent_space_plot:
        print_log('visulize latent embeddings via tsne ...', log)
        # only visualize latents from class 0 and class 1
        codes_embedded = TSNE(n_components=2).fit_transform(latents[class_inds])
        plot_2d_scatter(codes_embedded, misclass_array[class_inds], save_path=args.save_path, fig_name='feature_extractor_embedding')
        # return 
        


    #####################################################
    # # ---- Test Center: Select cases for testing ------
    print_log("\nTest Center Starts Working...", log)

    
    neg_case_indexes = []
    pos_case_indexes = []
    print_log("Test the labeled samples", log)
    labled_neg_indices = list(labeled_indices[np.nonzero(correct_array[labeled_indices])[0]])
    labled_pos_indices = list(labeled_indices[np.nonzero(misclass_array[labeled_indices])[0]])
    neg_case_indexes += labled_neg_indices
    pos_case_indexes += labled_pos_indices
    selected = set()

    # Iterative 
    print_log(' ** classification using GNN algorithm: {}'.format(args.graph_nn), log)
    print_log ("# of unlabeled test inputs: {}".format(len(unlabeled_indices)), log)

    # get classification result rank
    prob = softmax(logits, axis=1)
    confidence = np.sum(np.square(prob), axis=1)
    prob = torch.from_numpy(prob)

    # Step 2: rank samples
    # method1: apply label propagation algorithm
    if not args.graph_nn:
        if args.bf_mixed or args.learn_mixed: # mix
            _, output_distribution = propogate(latents, pos_case_indexes, neg_case_indexes)
            print_log("Mixed method enabled: combine gini and correlation based method", log)
            mix_rank_indicator = output_distribution[:, 1] * (1 - confidence) # the bigger, the more likely to be positive case
            ranked_indexes = np.argsort(mix_rank_indicator)[::-1] # cases having high positive probability are put in the front
        else: # correlation alone
            _, output_distribution = propogate(latents, pos_case_indexes, neg_case_indexes)
            ranked_indexes = np.argsort(output_distribution[:, 1])[::-1] # cases having high positive probability are put in the front
        return

    # method2: apply GNN classification algorithm
    old_time = time.time()
    hidden_dim = default_hyperparams[args.dataset]['gnn_hidden_dim']
    epochs = default_hyperparams[args.dataset]['gnn_train_epoch']
    if args.graph_nn:
        # create dataset
        print_log('Start runing GNN classification algorithm', log)

        # construct knn graph
        # original version
        # batch = torch.tensor([0 for _ in range(latents.shape[0])])
        # edge_index = knn_graph(torch.from_numpy(latents).float().to(device), 
        #                         batch=batch.to(device),
        #                         k=no_neighbors, 
        #                         cosine=True, loop=False)

        # print("edge_index: ", edge_index[:10])
        # new_time = time.time()
        # print_log("Finish calculate edge index, the shape is {}, time cost: {}".format(edge_index.shape, new_time-old_time), log)

        # approximate version
        x_l_indices = labeled_indices 
        x_u_indices = unlabeled_indices
        x_l = torch.from_numpy(latents[x_l_indices]).float().to(device)
        x_u = torch.from_numpy(latents[x_u_indices]).float().to(device)

        st = time.time()
        batch = torch.tensor([0 for _ in range(x_l.shape[0])]).to(device)
        edge_index_t = my_knn_graph(x_l, x_l, batch_x=batch, batch_y=batch, cosine=True, loop=False, k=no_neighbors)
        print ("l-2-l edge index: ", edge_index_t)
        #edge_index = torch.zeros_like(edge_index_t)
        new_edge_index_l0 = [x_l_indices[i] for i in list(edge_index_t[0])]
        new_edge_index_l1 = [x_l_indices[i] for i in list(edge_index_t[1])]
        l2l_edge_index = torch.tensor([new_edge_index_l0, new_edge_index_l1])
        print ("replaced edge index: ", l2l_edge_index)

        batch_x = torch.tensor([0 for _ in range(x_l.shape[0])]).to(device)
        batch_y = torch.tensor([0 for _ in range(x_u.shape[0])]).to(device)
        edge_index_t = my_knn_graph(x_l, x_u, batch_x=batch_x, batch_y=batch_y, cosine=True, loop=False, k=no_neighbors)
        print ("u-2-l edge index: ", edge_index_t)
        new_edge_index_l0 = [x_l_indices[i] for i in list(edge_index_t[0])]
        new_edge_index_l1 = [x_u_indices[i] for i in list(edge_index_t[1])]
        u2l_edge_index = torch.tensor([new_edge_index_l0, new_edge_index_l1])
        print ("replaced edge index: ", u2l_edge_index)

        edge_index = torch.cat([l2l_edge_index, u2l_edge_index], dim=1)
        print ("final edge_index, ", u2l_edge_index)
        print("Finish calculate edge index, the shape is {}, time cost: {:4f}".format(edge_index.shape, time.time()-st))
        # end approximation 
        
        edge_weight = torch.ones((edge_index.size(1), ),  device=edge_index.device)
        for i in range(edge_index.size(1)):
            edge_weight[i] = distance.cosine(latents[edge_index[0, i]], latents[edge_index[1, i]])
        print_log('Finish calculating edge weight. example of edge_weight: {}'.format(edge_weight[:10]), log)
        # print_log("time cost to calculate edge weight: {}".format(time.time()-new_time), log) 
        # return

        y = torch.zeros((latents.shape[0]), dtype=torch.long)
        imbalance_ratio = 1.*len(neg_case_indexes)/len(pos_case_indexes)
        print_log("neg:pos imbalance ratio: {}".format(imbalance_ratio), log)
                        
        # under-sampling
        # neg_case_indexes = random.sample(neg_case_indexes, len(pos_case_indexes))
        # class_weights = torch.tensor([1., 1.])

        # importance sampling
        class_weights = torch.tensor([1./(1+imbalance_ratio), 1.*imbalance_ratio/(1+imbalance_ratio)])
        # print_log("class_weights: {}".format(class_weights), log)

        y[pos_case_indexes] = 1 
        y[neg_case_indexes] = 0
        print ('positives: {}'.format(len(pos_case_indexes)))
        print ('negatives: {}'.format(len(neg_case_indexes)))

        dataset = Data(x=torch.from_numpy(latents).float(), y=y, edge_index=edge_index)
        dataset.edge_weight = edge_weight
        print ('example edge index: {}'.format(dataset.edge_index))
        print ('example edge index: max {} min {}'.format(dataset.edge_index[0].max(), dataset.edge_index[0].min()))
        # print ('example x: {}'.format(dataset.x[:10]))
        print ('example y: {}'.format(dataset.y.sum()))

        print_log("dataset info: {}".format(dataset), log)
        dataset.num_classes = y.max().item() + 1 
        print_log("**number of classes: {}".format(dataset.num_classes), log)

        # split train/val/test data
        dataset.train_mask = torch.zeros((latents.shape[0], ), dtype=torch.bool)
        dataset.val_mask = torch.zeros((latents.shape[0], ), dtype=torch.bool)
        dataset.test_mask = torch.zeros((latents.shape[0], ), dtype=torch.bool)

        labeled_list = list(neg_case_indexes + pos_case_indexes)
        random.shuffle(labeled_list)
        dataset.train_mask[labeled_list[:int(0.8*len(labeled_list))]] = True
        dataset.val_mask[labeled_list[int(0.8*len(labeled_list)):]] = True
        dataset.test_mask[list(set(range(latents.shape[0])) - set(labeled_list))] = True
        
        # logging 
        print_log("GNN training info", log)
        print_log("number of train cases: {}".format(dataset.train_mask.sum().item()), log)
        print_log("number of val cases: {}".format(dataset.val_mask.sum().item()), log)
        print_log("number of test cases: {}".format(dataset.test_mask.sum().item()), log)
        print_log("GNN input dimension: {}".format(dataset.num_node_features), log)
        print_log ("GNN output dimension: {}".format(dataset.num_classes), log)
        
        # build gnn model
        gcn_model   = GNNStack(input_dim=max(dataset.num_node_features, 1), 
                        hidden_dim=hidden_dim, 
                        output_dim=dataset.num_classes)                
        opt = optim.Adam(gcn_model.parameters(), lr=0.001, weight_decay=5e-4)

        # build mlp model 
        # if args.learn_mixed:
        print ('shape of sample embedding: ', hidden_dim+prob.shape[1]+1)
        mlp_model = MLP(hidden_dim+prob.shape[1]+1, 2).to(device)
        lc_optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        mlp_best_loss = 100
        mlp_model_path = os.path.join(args.save_path, 'mlp_model.ckpt')

        recorder = RecorderMeter(epochs)
        mlp_recorder = RecorderMeter(epochs)
        dynamic_dataset = copy.deepcopy(dataset)
        for e_iter in range(epochs):
            mlp_best_loss = train_graph(gcn_model, mlp_model, 
                                        confidence, prob, lc_optimizer, 
                                        mlp_best_loss, mlp_model_path, criterion, 
                                        opt=opt, 
                                        dataset=dataset, dynamic_dataset=dynamic_dataset,
                                        class_weights=class_weights, 
                                        recorder=recorder, mlp_recorder=mlp_recorder,
                                        epoch=e_iter, log=log, args=args)                
        recorder.plot_curve(os.path.join(args.save_path, 'gcn_train_curve.pdf'))
        mlp_recorder.plot_curve(os.path.join(args.save_path, 'mlp_train_curve.pdf'))

        # test
        # with torch.no_grad():
        #     emb, output_distribution = gcn_model_3(dynamic_dataset.x, dynamic_dataset.edge_index, edge_weight=dynamic_dataset.edge_weight)
        #     output_distribution  = torch.exp(output_distribution)
        #     # print_log("output distribution: {}".format(output_distribution.shape), log=log)
        # output_distribution = output_distribution.cpu().numpy()
        gcn_model.eval()
        with torch.no_grad():
            emb, output_distribution = gcn_model(dataset.x, dataset.edge_index, edge_weight=dataset.edge_weight)
            output_distribution  = torch.exp(output_distribution)
            # print_log("output distribution: {}".format(output_distribution.shape), log=log)
        output_distribution = output_distribution.cpu().numpy()

        if args.bf_mixed:
            # collaborative judgement of confidence and GNN 
            print_log("Mixed method enabled: combine gini and GNN based method", log)
            mix_rank_indicator = output_distribution[:, 1] * (1 - confidence) # the bigger, the more likely to be positive case
            ranked_indexes = np.argsort(mix_rank_indicator)[::-1].astype(np.int64) # cases having high positive probability are put in the front
        elif args.learn_mixed:
            print_log("Mixed method enabled: combine gini and correlation based method, learn based", log)
            print_log("dim: {}/{}/{}".format(emb.shape, torch.tensor(confidence).unsqueeze(1).shape, prob.shape), log)                    
            # visualize the learned embedding by gcn_model
            codes_embedded = TSNE(n_components=2).fit_transform(emb.detach().cpu()[class_inds])
            plot_2d_scatter(codes_embedded, misclass_array[class_inds], save_path=args.save_path, fig_name='gcn_embedding')
            
            sample_emd = torch.cat([emb.to(device), torch.tensor(confidence).unsqueeze(1).to(device), prob.to(device)], dim=1).type(torch.FloatTensor)

            # visualize the learned embedding by gcn_model
            if args.latent_space_plot:
                codes_embedded = TSNE(n_components=2).fit_transform(sample_emd.detach().cpu()[class_inds])
                plot_2d_scatter(codes_embedded, misclass_array[class_inds], save_path=args.save_path, fig_name='mix_embedding')                                                        
            
            # test
            # load best model from checkpoint
            mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
            outputs = mlp_model(sample_emd.to(device))
            output_distribution = F.softmax(outputs.cpu(), dim=1)
            # print (output_distribution)

            output_distribution = output_distribution.detach().numpy()
            ranked_indexes = np.argsort(output_distribution[:, 1])[::-1].astype(np.int64)
        else:
            ranked_indexes = np.argsort(output_distribution[:, 1])[::-1].astype(np.int64)


    index2select = [i for i in ranked_indexes if ((i not in selected) and (i not in labeled_indices))]
    print_log("ranked output distri for selection: {}".format(output_distribution[:, 1][index2select][:100]), log)

    
    for p_budget in p_budget_lst:
        print_log("\n ###### budget percent is {}% ######".format(p_budget), log)
        selected_temp = copy.deepcopy(selected)
        model2test_temp = copy.deepcopy(model2test)
        neg_case_indexes_temp = copy.deepcopy(neg_case_indexes)
        pos_case_indexes_temp = copy.deepcopy(pos_case_indexes)

        budget = int(p_budget*len(unlabeled_indices)/100.0)

        available_slots = budget - len(selected_temp)
        sel_indexes = index2select[:available_slots]
        sel_indexes = np.array(sel_indexes)

        # Step 3: test 
        selected_temp.update(set(sel_indexes))
        pos_count = misclass_array[sel_indexes].sum()
        neg_t =  list(sel_indexes[np.nonzero(correct_array[sel_indexes])[0]])
        pos_t =  list(sel_indexes[np.nonzero(misclass_array[sel_indexes])[0]])

        neg_case_indexes_temp += neg_t
        pos_case_indexes_temp += pos_t
        
        assert (len(neg_case_indexes_temp) == len(set(neg_case_indexes_temp))) # make sure no duplicated elements in the selected list
        assert (len(pos_case_indexes_temp) == len(set(pos_case_indexes_temp)))

        # logging
        print_log('==> total selected count.: {}'.format(len(selected_temp)), log)
        print_log('==> total pos count: {}'.format(pos_count), log)
        # print_log('==> current selected indexes: {}'.format(sel_indexes), log )
        print_log('    -- positive count: {}'.format(misclass_array[sel_indexes].sum()), log)
        print_log('    -- neg_case_indexes length: {}'.format(len(neg_t)), log)
        print_log('    -- pos_case_indexes length: {}'.format(len(pos_t)), log)


        # evaluation metric 1
        print_log("pos_count: {}".format(pos_count), log)
        print_log("total bug count: {}".format(misclass_array[unlabeled_indices].sum()), log)
        p_fault_detected = 100.0*pos_count/misclass_array[unlabeled_indices].sum()
        ideal_fault_detected = 100.0*budget/misclass_array[unlabeled_indices].sum()
        if ideal_fault_detected > 100.0:
            ideal_fault_detected = 100.00
        random_p_fault_detected = (100.0*budget/misclass_array[unlabeled_indices].shape[0])

        print_log("Model2test: {}".format(args.model2test_path), log)
        print_log("Model2Test Accuracy on labeled data: {}".format(100.0*correct_array[labeled_indices].sum()/misclass_array[labeled_indices].shape[0]), log)
        print_log("Total faults: {}".format(misclass_array[unlabeled_indices].sum()), log)
        print_log("Total test cases: {}".format(len(unlabeled_indices)), log)
        print_log("Percentage of fault detected: %s "%(p_fault_detected), log)
        print_log("Percentage of fault detected (random): %s "%(random_p_fault_detected), log)

        # output and logging
        out_file = os.path.join(args.save_path, 'gnn_result.csv' if args.graph_nn else 'lp_result.csv' )
        print_log ("writing output to csv file: {}".format(out_file), log)

        budget_lst.append(p_budget/100.0)
        pfd_lst.append(p_fault_detected/100.0)
        ideal_pfd_lst.append(ideal_fault_detected/100.0)

        apfd = 0
        ideal_apfd = 0
        if p_budget == 100:
            apfd = get_APFD(copy.copy(budget_lst), pfd_lst)
            ideal_apfd = get_APFD(budget_lst, ideal_pfd_lst)

        with open(out_file, 'a+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([ 
                            args.model2test_arch, 
                            args.model2test_path,
                            budget, 
                            p_budget,
                            args.sel_method, 
                            no_neighbors,
                            'FaultDetected',
                            p_fault_detected,
                            ideal_fault_detected,
                            random_p_fault_detected, 
                            'APFD', 
                            apfd,
                            ideal_apfd,
                            'TRC',
                            p_fault_detected/ideal_fault_detected
                            ])
        
        # overlap with gini
        conf_ranked_indexes = np.argsort(confidence)
        conf_index2select = [i for i in conf_ranked_indexes if i not in labeled_indices]
        confidence_selected = conf_index2select[:budget]
        
        overlap_ratio = 1.0*len(set(confidence_selected).intersection(set(selected_temp))) / len(confidence_selected)
        print_log ("overlap ratio: " + str(overlap_ratio), log)


    print_log('success!', log)
    log.close()

    return 



def plot_2d_scatter(codes_embedded, labels, save_path, fig_name, cmap=plt.get_cmap("seismic")):
    # visulize with tSNE
    # print('tsne plotting...')
    plt.figure()
    # colormap = np.array(['r', 'g', 'k', 'b'])
    # plt.scatter(codes_embedded[:, 0], codes_embedded[:, 1], s=1, c=labels, cmap=cmap, label='latent space visulization') #plot the latents of class 0 to class 19 (20 classes)
    plt.scatter(codes_embedded[:, 0][labels!=1], codes_embedded[:, 1][labels!=1], c='grey', s=0.5) #plot the latents of class 0 to class 19 (20 classes)
    plt.scatter(codes_embedded[:, 0][labels==1], codes_embedded[:, 1][labels==1], c='r', s=0.5) #plot the latents of class 0 to class 19 (20 classes)

    # plt.colorbar()
    plt.legend()
    # plt.title('Correct and Incorrect image latents')
    if not os.path.isdir('figs'):
        os.makedirs('figs')
    # plt.xticks(fontsize=14)
    plt.savefig(os.path.join(save_path, fig_name + '.pdf'))
    plt.close()


class selector():
    # select n latents out of all latents
    # return the selected latents index
    def __init__(self, latents, budget):
        self.latents = latents #numpy array
        self.budget = budget
        self.latent_len = latents.shape[0]
        assert (budget <= self.latent_len)

    def get_latents_len(self):
        return self.latent_len
    
    def _random_(self):
        random_index = np.random.choice(range(self.latent_len), self.budget, replace=False)
        selected_latents = self.latents[random_index]
        return selected_latents, random_index
    
    def _kmeans_(self):
        kmeans = KMeans(n_clusters=int(self.budget), random_state=0)
        kmeans = kmeans.fit(self.latents)
        centers = kmeans.cluster_centers_
        sel_index = np.zeros((self.budget,), dtype=int)
        for i in range(self.budget):
            rt = self.latents - centers[i] # calculate the distance between latents and the cluster centers
            # print_log(mu, centers[i], rt)
            sel_sim = np.linalg.norm(rt, axis=1) # select the latents closest to the cluster centers
            # print_log("L2 norm: ", sel_sim)
            sel_index[i] = np.argmin(sel_sim)
        
        selected_latents = self.latents[sel_index]
        return selected_latents, sel_index

    def sel(self, method):
        if method == 'random':
            return self._random_()
        elif method == 'kmeans':
            return self._kmeans_()

    def visualize(self, save_path):
        a, _ = self._random_()
        b, _ = self._kmeans_()
        latents = np.concatenate((a, b, self.latents), axis=0) # plot latents in one figure
        labels = np.zeros(2*self.budget + self.latent_len)  
        labels[:self.budget] = 2    # assign different latents to different labels
        labels[self.budget:2*self.budget] = 1
        
        codes_embedded = TSNE(n_components=2).fit_transform(latents) # tsne

        plot_2d_scatter(codes_embedded, labels, save_path=save_path, fig_name='comparison')
    


def test_acc(testset, model):
    batch_size = 1024
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    correct = 0
    total = 0
    model.eval()

    # test
    with torch.no_grad():
        for (inputs, labels) in tqdm(testloader):
            # print ("Test: {}/{}".format(batch_idx, int(len(testset)/batch_size)+1))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item() 
            total += labels.size(0)

    return 100.0*correct/total



def test(testset, model, num_classes=10):
    batch_size = 256
    testsize = len(testset)
    # print_log("test size %s"%testsize)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
    correct = 0
    total = 0
    model.eval()

    # test
    correct_array = np.zeros((testsize, ), dtype=int)
    logits = np.zeros((testsize, num_classes), dtype=float)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            # print ("Extracting features: {}/{}".format(batch_idx, int(len(testset)/batch_size)+1))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, pred = outputs.max(1)
            logits[(batch_idx)*batch_size: (batch_idx+1)*batch_size] = outputs.cpu().numpy()
            correct_array[(batch_idx)*batch_size: (batch_idx+1)*batch_size] = pred.eq(labels).cpu().numpy().astype(int)
            correct += pred.eq(labels).sum().item() 
            total += labels.size(0)

    return correct_array, logits 


def get_features(testset, model):
    batch_size= 256
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
    model.eval()

    # test
    feature_vector = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            if batch_idx % 5 == 0:
                print ("Extracting features: {}/{}".format(batch_idx, int(len(testset)/batch_size)+1))
            inputs = inputs.to(device)
            h = model(inputs)
            h = h.squeeze()
            h = h.detach()            
            feature_vector.extend(h.cpu().detach().numpy())
    
    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))

    return feature_vector




def get_latents(testset, encoder):
    # get latents of data, return numpy array
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
    encoder.eval()      
    data = next(iter(testloader))
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        mu, _ = encoder(inputs)
    mu = mu.cpu().data.numpy()
    return mu


def get_neighbor_index(seeds_index, latents):
    # calculate all distances to seed latent vectors
    # return the ranked indexes
    num_seeds = len(seeds_index)
    print("    *** number of seeds: ", num_seeds)
    neighbor_index = np.zeros((num_seeds, latents.shape[0]), dtype=int)
    for i in range(num_seeds): 
        dist = latents - latents[seeds_index[i]]
        sel_sim = np.linalg.norm(dist, axis=1) # calculate the L2 distances
        # print_log("L2 norm: ", sel_sim)
        neighbor_index[i] = np.argsort(sel_sim)   #sort by distance in asc. order     

    return neighbor_index


def retrain_model_under_test(net, sel_set, optimizer, criterion, train_set, train_transform):
    print ('classifier train_set length: {}'.format(len(train_set)))
    train_set_temp = torch.utils.data.ConcatDataset([train_set, sel_set])
    recover_loader = torch.utils.data.DataLoader(train_set_temp, batch_size=128, shuffle=True)

    net.train()
    total = 0
    correct = 0
    train_loss = 0
    for batch_idx, data in enumerate(recover_loader):
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs, _ = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)

        correct += predicted.eq(labels).sum().item()
        train_loss += loss.item()
        total += labels.size(0)

        print ("Train Batch [%d/%d]"%(batch_idx, len(recover_loader)), 'Loss: %.3f | Acc: %.3f(%d/%d)'
                                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return net


def save_images(datasets, save_array, save_path, transform):
    # transfrom_para: (std, mean)
    # datasets: all images
    # save_array: indexes of images to be saved
    imgs = []
    save_index = np.nonzero(save_array)[0]
    for i in range(save_array.sum()):
        img, _ = datasets[save_index[i]]
        # mis_imgs.append(data.recover_image(img.unsqueeze(0), std, mean))
        imgs.append((img.unsqueeze(0)))

    imgs2save = torch.cat(imgs)
    save_image(data.recover_image(imgs2save.cpu(), transform[0], transform[1]),
                    save_path, nrow=20)
    

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels, data_mask):
        'Initialization'
        self.masked_data = data[data_mask]
        self.masked_label = labels[data_mask]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.masked_label)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.masked_data[index]
        y = self.masked_label[index]

        return x, y


def set_weights_for_classes(dataset, weight_per_class):                                                                           
    # weight_per_class = np.random.rand(nclasses)
    print ("weight per class: ", weight_per_class)                                                 
    weight = [0] * len(dataset)     
    for idx, (img, label) in enumerate(dataset):    
        # print ('assign weigh {} / {}'.format(idx, len(dataset)))                                      
        weight[idx] = weight_per_class[label]                                  
    return weight  


def my_knn_graph(x: torch.Tensor, y:torch.Tensor, k: int, 
                batch_x: Optional[torch.Tensor] = None,
                batch_y: Optional[torch.Tensor] = None,
                loop: bool = False, flow: str = 'source_to_target',
                cosine: bool = False, num_workers: int = 1) -> torch.Tensor:

    assert flow in ['source_to_target', 'target_to_source']
    # Finds for each element in :obj:`y` the :obj:`k` nearest points in obj:`x`.
    edge_index = knn(x, y, k if loop else k + 1, batch_x, batch_y, cosine,
                     num_workers)

    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


class MLP(nn.Module):
    def __init__(self, num_ftrs, out_dim):
         super(MLP, self).__init__()
         self.l1 = nn.Linear(num_ftrs, num_ftrs)
         self.l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), 
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.2
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        return pyg_nn.GCNConv(input_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        x, edge_index = x, edge_index
       
        for i in range(self.num_layers):    
            x = self.convs[i](x, edge_index, edge_weight)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)
        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label, weight):
        return F.nll_loss(input=pred, target=label, weight=weight)


def train_graph(gcn_model, mlp_model, confidence, prob, lc_optimizer, mlp_best_loss, mlp_model_path, criterion,
            opt, dataset, dynamic_dataset, class_weights, recorder, mlp_recorder, log, epoch, args):
    global no_neighbors

    gcn_model = gcn_model.to(device) 
    dataset.x, dataset.edge_index = dataset.x.to(device), dataset.edge_index.to(device)
    dataset.y = dataset.y.to(device)
    dataset.edge_weight = dataset.edge_weight.to(device)
    dataset.train_mask = dataset.train_mask.to(device)
    dataset.val_mask = dataset.val_mask.to(device)
    dataset.test_mask = dataset.test_mask.to(device)
    batch = torch.tensor([0 for _ in range(dataset.x.shape[0])]).to(device)

    # train
    gcn_model.train()
    t = time.time()
    gcn_correct = 0
    total = 0
    gcn_train_loss = 0
    opt.zero_grad()
    
    mlp_start_epoch = 450

    # gcn 1 loss
    # if epoch < 150:
    emb, pred = gcn_model(dataset.x, dataset.edge_index, edge_weight=dataset.edge_weight)

    label = dataset.y[dataset.train_mask]
    pred = pred[dataset.train_mask]

    gcn_loss = gcn_model.loss(pred, label, class_weights.to(device))
    # gcn train acc
    gcn_train_loss += gcn_loss.item() 
    gcn_correct +=  pred.argmax(1).eq(label).sum().item() 
    total += len(label)
    
    gcn_train_acc = 100.0*gcn_correct / total

    # mlp loss
    if args.learn_mixed and epoch>=mlp_start_epoch:
        
        mlp_train_correct = 0
        mlp_train_loss = 0

        sample_emd = torch.cat([emb.detach(), torch.tensor(confidence).unsqueeze(1).to(device), 
                            prob.to(device)], dim=1).type(torch.FloatTensor)
        train_set = Dataset(sample_emd, dataset.y, dataset.train_mask)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128)
        for local_data, local_labels in train_loader:
            lc_optimizer.zero_grad()
            outputs = mlp_model(local_data.to(device))
            mlp_loss = criterion(outputs, local_labels.to(device))
            mlp_train_loss += mlp_loss.cpu().item()
            _, predicted = outputs.max(axis=1)
            mlp_train_correct += predicted.eq(local_labels).sum().item()
            # train gcn and mlp together
            # loss = gcn_loss + mlp_loss
            # loss.backward(retain_graph=True)
            # lc_optimizer.step()
            # outputs = None
            # train mlp alone
            loss = mlp_loss
            loss.backward()
            lc_optimizer.step()
        mlp_train_acc = 100.0*mlp_train_correct / total
        # train gcn and mlp together
        # opt.step()
    else:
        loss = gcn_loss
        loss.backward()
        opt.step()


    # val gcn
    gcn_val_acc, gcn_precision, gcn_recall, gcn_f1_score = test_graph(dataset, gcn_model, is_validation=True)

    if epoch < mlp_start_epoch:
        print('GCN MODEL Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(gcn_train_loss),
            'acc_train: {:.4f}'.format(gcn_train_acc),
            'acc_val: {:.4f}'.format(gcn_val_acc),
            'precision: {:.4f}'.format(gcn_precision),
            'recall: {:.4f}'.format(gcn_recall),
            'f1_score: {:.4f}'.format(gcn_f1_score),
            'time: {:.4f}s'.format(time.time() - t))
        recorder.update(epoch, 
                        train_loss=gcn_train_loss, 
                        train_acc=gcn_train_acc, 
                        val_loss=0, 
                        val_acc=gcn_val_acc)

    # val mlp
    if args.learn_mixed and epoch>=mlp_start_epoch:
        mlp_model.eval()
        mlp_val_loss = 0
        # total = 0
        # mlp_val_correct = 0
        sample_emd = torch.cat([emb[dataset.val_mask], torch.tensor(confidence).unsqueeze(1)[dataset.val_mask].to(device), 
                            prob[dataset.val_mask].to(device)], dim=1).type(torch.FloatTensor)
        local_batch, local_labels = sample_emd, dataset.y[dataset.val_mask]
        outputs = mlp_model(local_batch.to(device))
        loss = criterion(outputs, local_labels)
        mlp_val_loss += loss.cpu().item()
            
        _, predicted = outputs.max(1)
        mlp_val_correct = predicted.eq(local_labels).sum().item()
        total = local_labels.size(0)
        mlp_val_acc = 100.0*mlp_val_correct / total

        if mlp_val_loss < mlp_best_loss:
            mlp_best_loss = mlp_val_loss
            # save checkpoint 
            print ('saving checkpoint...')
            torch.save(mlp_model.state_dict(), mlp_model_path)
        
        print('MLP MODEL Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(mlp_train_loss),
            'acc_train: {:.4f}'.format(mlp_train_acc),
            'acc_val: {:.4f}'.format(mlp_val_acc),
            'time: {:.4f}s'.format(time.time() - t))
        
        mlp_recorder.update(int((epoch-mlp_start_epoch)), 
                    train_loss=mlp_train_loss, 
                    train_acc=mlp_train_acc, 
                    val_loss=mlp_val_loss, 
                    val_acc=mlp_val_acc)

    return mlp_best_loss
    

def test_graph(dataset, model, is_validation=False):
    global no_neighbors
    model.eval()
    dataset = dataset.to(device)
    correct = 0
    with torch.no_grad():
        emb, pred = model(dataset.x, dataset.edge_index, dataset.edge_weight)
        pred = pred.argmax(dim=1)

        mask = dataset.val_mask if is_validation else dataset.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = dataset.y[mask]
        # print ("val label ratio: ", np.array(label.cpu()==0).sum()/np.array(label.cpu()==1).sum())
            
        correct += pred.eq(label).sum().item()
    
    total = len(label)
    
    precision = sklearn.metrics.precision_score(y_true=label.cpu(), y_pred=pred.cpu(), zero_division='warn')
    recall = sklearn.metrics.recall_score(y_true=label.cpu(), y_pred=pred.cpu())
    f1_score = 2*precision*recall/(precision+recall+1e-6)
    if is_validation == False:
       print ("consufion matrix:\n ", sklearn.metrics.confusion_matrix(y_true=label.cpu(), y_pred=pred.cpu()))
    return 100.*correct / total, precision, recall, f1_score


def get_APFD(budget_lst, pfd_lst):
    # budget_list: value 0-1
    # pfd_list: value 0-1
    # assert (budget_lst[0] == 0.0 & pfd_lst[0] == 0.0 & budget_lst[-1] == 1.0 & pfd_lst[-1] == 1.0)
    # print ('budget list, pfd_lst: ', budget_lst, pfd_lst)
    apfd = 0
    for i in range(len(budget_lst)):
        if i == 0:
            continue
        else:
            area_temp = (budget_lst[i] - budget_lst[i-1]) * (pfd_lst[i] - pfd_lst[i-1]) / 2 \
                        + (budget_lst[i] - budget_lst[i-1]) * pfd_lst[i-1]
    
        apfd += area_temp
    return apfd 


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


if __name__ == '__main__':
    main()
