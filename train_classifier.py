import argparse
import os, sys 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

import random
import numpy as np
import mymodels
import utils
import data     
import ast 
from datetime import date
import copy

today = date.today()
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print ("Use device: ", device)

data_segmentation = {
    'emnist': { 'TC_end':5e5, 'DC_L_end':5e5+1e4,'DC_U_end':5e5+1e4+2e5}
}

img_channels = {
    'cifar10':3,
    'svhn': 3,
    'emnist': 1,
    'stl10': 3
}

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_acc = 0.0
best_acc = 0.0

# train   
def train(epoch, net, trainloader, criterion, optimizer):
    global train_acc
    net.train()
    total = 0
    correct = 0
    train_loss = 0
    
    for batch_idx, data in enumerate(trainloader):
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

        print ("Epoch [%d] Train Batch [%d/%d]"%(epoch, batch_idx, len(trainloader)), 'Loss: %.3f | Acc: %.3f(%d/%d)'
                                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_acc = 100.*correct/total


# test
def test(epoch, net, testloader, criterion, model_save_path):
    global best_acc
    net.eval()
    total = 0
    correct = 0
    test_loss = 0
    for batch_idx, data in enumerate(testloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, _ = net(inputs)
        loss = criterion(outputs, labels)

        _, pred = outputs.max(1)
        total += labels.size(0)
        test_loss += loss.item()

        correct += pred.eq(labels).sum().item()

        print ("Test Batch [%d/%d]"%(batch_idx, len(testloader)), 'Loss: %.3f | Acc: %.3f(%d/%d)'
                                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    test_acc = 100.*correct/total
    if test_acc > best_acc and (model_save_path is not None):
        print('Saving..')
        if device == 'cuda':
            state = net.module.state_dict()
        else:
            state = net.state_dict()
        state = {
            'net': state,
            'acc': test_acc,
            'epoch': epoch,
        }
        torch.save(state, model_save_path)
        best_acc = test_acc
    return test_acc


def set_weights_for_classes(dataset, weight_per_class):                                                                           
    # weight_per_class = np.random.rand(nclasses)
    print ("weight per class: ", weight_per_class)                                                 
    weight = [0] * len(dataset)     
    for idx, (img, label) in enumerate(dataset):    
        # print ('assign weigh {} / {}'.format(idx, len(dataset)))                                      
        weight[idx] = weight_per_class[label]                                  
    return weight  


def main():
    global data_segmentation

    parser = argparse.ArgumentParser("Train a classifier.")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="supportive dataset: (cifar10, stl10)")
    parser.add_argument("--model", type=str, default=None, 
                        help="(resnet18, resnet50)")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    # parser.add_argument("--img_size", type=int, default=224, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=200, 
                        help="number of epochs of training")
    parser.add_argument("--class_weight", type=int, default=0, 
                         help='choose from three different settings: all ones, random 1, random 2')
    parser.add_argument("--pretrained", action="store_true", default=False, 
                        help="if we are to use the imagenet pretrained model or not")
    parser.add_argument("--shadow", action='store_true', help='train a shadow classifier with test set')
    parser.add_argument("--data_root", type=str, help="dataset directory")
    parser.add_argument("--save_path", type=str, help="log save directory")
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')

    args = parser.parse_args()
    print (args)

    # Give a random seed if no manual configuration
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.manualSeed)

    print (args.manualSeed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                        'log_train_classifier_seed_{}.txt'.format(args.manualSeed)), 'w')
                        
    data_dir = os.path.join(args.data_root, args.dataset)

    # num of classes
    num_class_config = {'stl10':10, 'cifar10':10, 'cifar100':100, 'gtsrb':43, 'tinyimagenet':200, 'svhn':10, 'emnist':47}
    num_classes = num_class_config[args.dataset]

    # weight per class
    if args.dataset == 'cifar10' or args.dataset == 'svhn' or args.dataset == 'stl10':
        weight_per_class_1ist = [np.ones(num_classes), 
                             np.array([0.77132064, 0.02075195, 0.63364823, 0.74880388, 0.49850701,
                                    0.22479665, 0.19806286, 0.76053071, 0.16911084, 0.08833981]), 
                             np.array([0.68535982, 0.95339335, 0.00394827, 0.51219226, 0.81262096,
                                    0.61252607, 0.72175532, 0.29187607, 0.91777412, 0.71457578])]
    elif args.dataset == 'emnist':
        weight_per_class_1ist = [np.ones(num_classes),
                                np.array([0.01126189, 0.91091961, 0.42194606, 0.46959393, 0.95206712,
                                        0.54583793, 0.66975958, 0.61898302, 0.05806656, 0.190098  ,
                                        0.84301274, 0.38444687, 0.54545051, 0.24376026, 0.03185426,
                                        0.32539058, 0.31705925, 0.31152293, 0.15854179, 0.93160301,
                                        0.56039436, 0.40209926, 0.70893203, 0.58077904, 0.82970958,
                                        0.13559519, 0.92300047, 0.99839883, 0.27749007, 0.86684817,
                                        0.52858135, 0.36618893, 0.91003319, 0.39327373, 0.87875105,
                                        0.06459116, 0.28789443, 0.14246855, 0.73571405, 0.21959115,
                                        0.38249527, 0.46639426, 0.26012537, 0.78598211, 0.29052295,
                                        0.97294385, 0.17234997]),
                                np.array([0.18563999, 0.43811132, 0.29870295, 0.24753067, 0.73752132,
                                        0.27903653, 0.49742426, 0.81565799, 0.98164481, 0.18600724,
                                        0.40306355, 0.15192068, 0.30726325, 0.89710922, 0.86371842,
                                        0.83078786, 0.65140476, 0.78072694, 0.1998836 , 0.23113152,
                                        0.03963373, 0.23156082, 0.88540162, 0.4110333 , 0.61840306,
                                        0.83058136, 0.5463734 , 0.47666202, 0.0396291 , 0.97546816,
                                        0.97476249, 0.87698951, 0.05907245, 0.46710997, 0.48639676,
                                        0.87498038, 0.44006725, 0.21709722, 0.51453244, 0.19790319,
                                        0.63053556, 0.44729439, 0.11430839, 0.8439266 , 0.16758325,
                                        0.77483716, 0.33671929])
                                ]
    else:
        raise ValueError("Dataset not supported by current version")

    weight_per_class = weight_per_class_1ist[args.class_weight]

    # ---- load datasets -----
    if args.dataset == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        val_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        
        # train and val set for classifier are both coming from the offical train set 
        train_set = torchvision.datasets.SVHN(root=data_dir, 
                                                split = 'train',
                                                transform=train_transform, 
                                                download=True)        
        val_set = torchvision.datasets.SVHN(root=data_dir, 
                                                split='train',
                                                transform=val_transform, 
                                                download=True)
        # print (min(train_set.labels), max(train_set.labels))
        num_train = len(train_set)
        indices = list(range(num_train))
        train_idx, valid_idx = indices[:40000], indices[40000:50000]

        train_set = torch.utils.data.Subset(train_set, train_idx)
        val_set = torch.utils.data.Subset(val_set, valid_idx)
        print (len(train_set), len(val_set))

        weights = set_weights_for_classes(train_set, weight_per_class)
        train_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        train_index = list(train_index)

        targets = [i[1] for i in train_set]
        plt.figure()
        plt.hist([targets[i] for i in train_index])
        plt.savefig(os.path.join(args.save_path, 'trainset_distribution.png'))
        
        train_set = torch.utils.data.Subset(train_set, train_index)
        # print ("trainset information: ", len(train_set), train_set[0])
        weights = set_weights_for_classes(val_set, weight_per_class)
        val_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        val_index = list(val_index)

        targets = [i[1] for i in val_set]
        plt.figure()
        plt.hist([targets[i] for i in val_index])
        plt.savefig(os.path.join(args.save_path, 'valset_distribution.png'))

        val_set = torch.utils.data.Subset(val_set, val_index) 
        print_log("dataset segmentation, train/val {}/{}".format(len(train_set), len(val_set)), log)

        train_loader = torch.utils.data.DataLoader(train_set, 
                                    shuffle=True,
                                    batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
                                    val_set, 
                                    shuffle=False, batch_size=args.batch_size)

         
    elif args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
             
        train_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                train=True, 
                                                transform=train_transform, 
                                                download=True)        
        val_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                train=True, 
                                                transform=val_transform, 
                                                download=True)
        print (train_set[0])
        num_train = len(train_set)
        indices = list(range(num_train))
        train_idx, valid_idx = indices[:15000], indices[15000:20000]

        train_set = torch.utils.data.Subset(train_set, train_idx)
        val_set = torch.utils.data.Subset(val_set, valid_idx)

        weights = set_weights_for_classes(train_set, weight_per_class)
        train_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        train_index = list(train_index)

        targets = [i[1] for i in train_set]
        plt.hist([targets[i] for i in train_index])
        plt.savefig('trainset_distribution.png')

        train_set = torch.utils.data.Subset(train_set, train_index)
        print ("trainset information: ", len(train_set), train_set[0])
        weights = set_weights_for_classes(val_set, weight_per_class)
        val_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        val_index = list(val_index)

        targets = [i[1] for i in val_set]
        plt.hist([targets[i] for i in val_index])
        plt.savefig('valset_distribution.png')

        val_set = torch.utils.data.Subset(val_set, val_index) 
        print_log("dataset segmentation, train/val {}/{}".format(len(train_set), len(val_set)), log)

        train_loader = torch.utils.data.DataLoader(train_set, 
                                    shuffle=True,
                                    batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
                                    val_set, 
                                    shuffle=False, batch_size=args.batch_size)

    elif args.dataset == 'emnist':
        data_seg = data_segmentation[args.dataset]
        mean, std = (0.5), (0.5)
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor()])
             
        train_set = torchvision.datasets.EMNIST(root=data_dir, 
                                                split='bymerge', 
                                                train=True,
                                                transform=train_transform, 
                                                download=False)        
        val_set = torchvision.datasets.EMNIST(root=data_dir, 
                                                split='bymerge', 
                                                train=True,
                                                transform=val_transform, 
                                                download=False)
        print ("official data train/", len(train_set))
        num_train = len(train_set)
        indices = list(range(num_train))
        train_idx, valid_idx = indices[:int(data_seg['TC_end']*0.8)], indices[int(data_seg['TC_end']*0.8):int(data_seg['TC_end'])]

        print ("targets: ", train_set.class_to_idx)
        train_set = torch.utils.data.Subset(train_set, train_idx)
        val_set = torch.utils.data.Subset(val_set, valid_idx)

        weights = set_weights_for_classes(train_set, weight_per_class)

        
        print (len(train_set), len(val_set), len(weights))
        # print (weights)
        train_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        print (train_index)
        train_index = list(train_index)

        targets = [i[1] for i in train_set]
        plt.hist([targets[i] for i in train_index])
        plt.savefig('trainset_distribution.png')

        train_set = torch.utils.data.Subset(train_set, train_index)
        print ("trainset information: ", len(train_set), train_set[0])
        weights = set_weights_for_classes(val_set, weight_per_class)
        val_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        val_index = list(val_index)

        targets = [i[1] for i in val_set]
        plt.hist([targets[i] for i in val_index])
        plt.savefig('valset_distribution.png')

        val_set = torch.utils.data.Subset(val_set, val_index) 
        print_log("dataset segmentation, train/val {}/{}".format(len(train_set), len(val_set)), log)

        train_loader = torch.utils.data.DataLoader(train_set, 
                                    shuffle=True,
                                    batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
                                    val_set, 
                                    shuffle=False, batch_size=args.batch_size)
    elif args.dataset == 'stl10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val_transform = transforms.Compose(
            [transforms.Resize((96, 96)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
             
        train_set = torchvision.datasets.STL10(root=data_dir, 
                                                split='train', 
                                                transform=train_transform, 
                                                download=True)        
        val_set = torchvision.datasets.STL10(root=data_dir, 
                                                split='train', 
                                                transform=val_transform, 
                                                download=True)
        print (train_set[0])
        num_train = len(train_set)
        indices = list(range(num_train))
        train_idx, valid_idx = indices[:4000], indices[4000:5000]

        train_set = torch.utils.data.Subset(train_set, train_idx)
        val_set = torch.utils.data.Subset(val_set, valid_idx)

        weights = set_weights_for_classes(train_set, weight_per_class)
        train_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.9*len(weights)))
        train_index = list(train_index)

        targets = [i[1] for i in train_set]
        plt.hist([targets[i] for i in train_index])
        plt.savefig('trainset_distribution.png')

        train_set = torch.utils.data.Subset(train_set, train_index)
        print ("trainset information: ", len(train_set), train_set[0])
        weights = set_weights_for_classes(val_set, weight_per_class)
        val_index = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.DoubleTensor(weights), 
                                                                    replacement=False,
                                                                    num_samples=int(0.75*len(weights)))
        val_index = list(val_index)

        targets = [i[1] for i in val_set]
        plt.hist([targets[i] for i in val_index])
        plt.savefig('valset_distribution.png')

        val_set = torch.utils.data.Subset(val_set, val_index) 
        print_log("dataset segmentation, train/val {}/{}".format(len(train_set), len(val_set)), log)

        train_loader = torch.utils.data.DataLoader(train_set, 
                                    shuffle=True,
                                    batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
                                    val_set, 
                                    shuffle=False, batch_size=args.batch_size)

    else:
        print_log ("Not valid datasets inputs, the available choice is stl10, imagenet.", log)
    

    # save the biased sample index 
    if not args.shadow:
        dataset_save_path = os.path.join('./checkpoint', 
                                            args.dataset, 
                                            'ckpt_bias', 
                                            'biased_dataset',
                                            str(args.class_weight))
        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)
        np.save(os.path.join(dataset_save_path, 'train.npy'), np.array(train_index))
        np.save(os.path.join(dataset_save_path, 'val.npy'), np.array(val_index))

    # ---- create model -----
    model_names = [name for name in mymodels.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(mymodels.__dict__[name])]
    
    print ('available models: ', model_names)
    print ("current model: ", args.model)
    
    net = mymodels.__dict__[args.model](channels=img_channels[args.dataset], num_classes=num_classes).to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
    print (net)

    # ----- Train classifer  ------
    # criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if not args.shadow:
        # IP vendor train the classifer
        model_save_path = os.path.join('./checkpoint', args.dataset, 'ckpt_bias')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        for epoch in range(args.n_epochs):
            print(
            '\n==>> [Epoch={:03d}/{:03d}] '.format(epoch, args.n_epochs) \
            + ' [Best : Accuracy={:.2f}]'.format(best_acc ))

            train(epoch, net, train_loader, criterion, optimizer)
            test(epoch, net, valid_loader, criterion, os.path.join(model_save_path, args.model + '_' + str(args.class_weight) +'_b.t7'))
        
        print_log('save model of ip vendor to' + model_save_path, log)
    else: # shadow is true
        # shadow classifier for test center
        # split labeled data from unlabeled data
        if args.dataset == 'cifar10':
            # re-allocate the dataset
            train_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                        train=True, 
                                        transform=val_transform, 
                                        download=True)
            test_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                        train=False, 
                                        transform=val_transform, 
                                        download=True)
            new_test_set = torch.utils.data.ConcatDataset([train_set, test_set])
            
            num_train = len(new_test_set)
            indices = list(range(num_train))
            labeled_indices = indices[21000:21000+int(39000*0.2)]
            shadow_train_idx, shadow_test_idx = indices[21000: 21000 + int(len(labeled_indices) * 0.8)], indices[21000 + int(len(labeled_indices) * 0.8): 21000+int(len(labeled_indices))]
            
            shadow_train_set = torch.utils.data.Subset(new_test_set, shadow_train_idx)
            shadow_test_set = torch.utils.data.Subset(copy.deepcopy(new_test_set), shadow_test_idx)
            
            for i in range(2): # there are two datasets in concatDataset
                shadow_train_set.dataset.datasets[i].transform = train_transform
                shadow_test_set.dataset.datasets[i].transform = val_transform
                print (shadow_train_set.dataset.datasets[i].transform)
                print (shadow_test_set.dataset.datasets[i].transform)
        else:
            raise ValueError('dataset must be cifar10')

        # plot
        targets = [i[1] for i in shadow_train_set]
        plt.figure()
        plt.hist(targets)
        plt.savefig('shadow_trainset_distribution.png')
        
        plt.figure()
        targets = [i[1] for i in shadow_test_set]
        plt.hist(targets)
        plt.savefig('shadow_valset_distribution.png')

        print_log("data to train shadow model: train/val = {}/{}".format(len(shadow_train_set), len(shadow_test_set)), log)
        
        shadow_train_loader = torch.utils.data.DataLoader(shadow_train_set, batch_size=args.batch_size,
                                            shuffle=True, num_workers=16)
        shadow_test_loader = torch.utils.data.DataLoader(shadow_test_set, batch_size=args.batch_size,
                                            shuffle=False, num_workers=16)

        model_save_path = os.path.join('./checkpoint', args.dataset, 'shadow')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        for epoch in range(args.n_epochs):
            train(epoch, net, shadow_train_loader, criterion, optimizer)
            test(epoch, net, shadow_test_loader, criterion, 
                        os.path.join(model_save_path, args.model + '_' + str(len(shadow_train_set)) + '_' + str(len(shadow_test_set)) + '.t7'))
        print_log('save shadow model to' + model_save_path, log)



def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


if __name__ == "__main__":
    main()