import os 
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import data_aug
import logging
from torch.utils.data.sampler import SubsetRandomSampler

# for reproducibility
torch.manual_seed(0)    
np.random.seed(0)

class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


def set_weights_for_classes(dataset, weight_per_class):                                                                           
    # weight_per_class = np.random.rand(nclasses)
    print ("weight per class: ", weight_per_class)                                                 
    weight = [0] * len(dataset)     
    for idx, (img, label) in enumerate(dataset):                                          
        weight[idx] = weight_per_class[label]                                  
    return weight  


def get_imagenet_dataloader(dataset, img_size, batch_size, contrastive=False):
    print ('Getting imagenet loader...')
    data_dir = os.path.join('/research/dept2/yuli/datasets/imagenet', dataset)
    # get data_dir
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if dataset=='tinyimagenet_data':
        # oringinally its 64*64 in size
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                        for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=1)
                        for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    print ("dataset size: train {} test {}".format(dataset_sizes['train'], dataset_sizes['test']))

    return image_datasets, dataloaders


class STL10_data_loader():
    def __init__(self, data_dir, img_size=96, batch_size=64, contrastive=False):
        logging.info("Initializing a STL dataloader")
        self.mean, self.std = get_std_and_mean('stl10')
        self.img_size = img_size
        self.batch_size = batch_size
        self.contrastive = contrastive

        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.data_transforms ={
            'train': transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(96),
                        transforms.Resize(img_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(self.mean, self.std),
                ]),

            'test': transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize(self.mean, self.std),
                    ]),
            'augmentation': transforms.Compose([transforms.RandomResizedCrop(size=img_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.5),
                                              transforms.RandomGrayscale(p=1),
                                              data_aug.GaussianBlur(kernel_size=int(0.1 * img_size + 1)),
                                              transforms.ToTensor()]),
                            
            }

        if self.contrastive is True:
            # image pairs (x_i, x_j), y
            self.image_datasets = {x: torchvision.datasets.STL10(root=os.path.join(data_dir, x), split=x,
                                                transform=SimCLRDataTransform(self.data_transforms['augmentation']), download=True)
                                                for x in ['train', 'test']}
        else:
            # image and label (x, y)
            self.image_datasets = {x: torchvision.datasets.STL10(root=os.path.join(data_dir, x), split=x,
                                            transform=self.data_transforms[x], download=True)
                                            for x in ['train', 'test']}
            
        # split test dataset into val (IP cpmpany) and test (Test Center)
        num_test = len(self.image_datasets['test'])
        indices = list(range(num_test))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * num_test))
        test_data_idx, val_data_idx = indices[split:], indices[:split] # val: 20%, test: 80%

        self.train_set = self.image_datasets['train']
        self.val_set = torch.utils.data.Subset(self.image_datasets['test'], val_data_idx)
        
        self.test_set = torch.utils.data.Subset(self.image_datasets['test'], test_data_idx)
        self.test_sampler = SubsetRandomSampler(test_data_idx)
        
        # count number of sampels in val_set
        count = [0]*10
        for i in range(len(self.val_set)):
            _, label = self.val_set[i]
            count[label] += 1

        # logging.info("num of total test cases: %s, \nnumber of sampels for each class\
        #                     in val set: %s"%(num_test, str(count)))
        # logging.info("\nNum. train data: %s \n Num. test data: %s\
        #                 \n Num. val data: %s"%(len(self.train_set), len(self.test_set), 
        #                 len(self.val_set)))

        
    def get_train_loader(self, weight_per_class=np.ones((10))):
        print ('Getting STL10 train loader of IP company...')
        # logging.info("Getting STL10 train loader of IP company...")

        # select biased training dataset
        weights = set_weights_for_classes(self.train_set, weight_per_class)
        weights = torch.DoubleTensor(weights) 
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        train_batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)

        train_loader = torch.utils.data.DataLoader(self.train_set, 
                                                    batch_sampler=train_batch_sampler, num_workers=16)
        return self.train_set, train_loader


    def get_val_loader(self, weight_per_class=np.ones((10))):
        print ('Getting STL10 val loader of IP company...')
        # logging.info("Getting STL10 val loader of IP company...")

        # select biased training dataset
        weights = set_weights_for_classes(self.val_set, weight_per_class)
        weights = torch.DoubleTensor(weights) 
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        val_batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)

        val_loader = torch.utils.data.DataLoader(self.val_set,
                                                 batch_sampler=val_batch_sampler, num_workers=16)
        return self.val_set, val_loader


    def get_test_loader(self):
        print ('Getting STL10 Dataset of Test Center...')
        # logging.info("Getting STL10 test loader of IP company...")

        # test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
        #                                     shuffle=False, num_workers=16) # set shuffle to True because we may use test loader to train classifier
        test_loader = torch.utils.data.DataLoader(self.image_datasets['test'], 
                                                    batch_size=self.batch_size,
                                                    sampler=self.test_sampler,
                                                    drop_last=True,
                                                    shuffle=False, num_workers=16) # set shuffle to True because we may use test loader to train classifier
        
        return self.test_set, test_loader



# def get_stl10_dataloader(dataset='stl10', img_size=96, batch_size=64, contrastive=False, bias=False, weight_per_class=np.ones((10))):
#     # originally its 96*96 in size 
#     print ('Getting STL10 loader...')
#     mean, std = get_std_and_mean('stl10')
#     data_dir = os.path.join('/research/dept2/yuli/datasets', dataset)
#     if not os.path.isdir(data_dir):
#         os.mkdir(data_dir)

#     if contrastive == False:

#         data_transforms ={
#             'train': transforms.Compose([
#                         transforms.Pad(4),
#                         transforms.RandomCrop(96),
#                         transforms.Resize(img_size),
#                         transforms.RandomHorizontalFlip(),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean, std),
#                 ]),

#             'test': transforms.Compose([
#                         transforms.Resize(img_size),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean, std),
#                     ])
#         }
#         image_datasets = {x: torchvision.datasets.STL10(root=os.path.join(data_dir, x), split=x,
#                                                 transform=data_transforms[x], download=True)
#                             for x in ['train', 'test']}
        
#         # create bias dataset with different weights for each class
#         if bias == True:
#             # to select randomly biased training dataset
#             weights = set_weights_for_classes(image_datasets['train'], weight_per_class)
#             weights = torch.DoubleTensor(weights) 
#             sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
#             train_batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
            
#             # split test dataset to two_halves
#             num_test = len(image_datasets['test'])
#             indices = list(range(num_test))
#             np.random.shuffle(indices)
#             split = int(np.floor(0.2 * num_test))
#             test_data_idx, val_data_idx = indices[split:], indices[:split] # val: 20%, test: 80%

#             val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_data_idx)
#             val_batch_sampler = torch.utils.data.sampler.BatchSampler(val_sampler, batch_size=batch_size, drop_last=False)
            
#             test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_data_idx)
#             test_batch_sampler = torch.utils.data.sampler.BatchSampler(test_sampler, batch_size=batch_size, drop_last=False)

#             dataloaders = { 'train': torch.utils.data.DataLoader(image_datasets['train'], 
#                                             batch_sampler=train_batch_sampler, num_workers=1),
#                             'val': torch.utils.data.DataLoader(image_datasets['test'], 
#                                             batch_sampler=val_batch_sampler, num_workers=1),
#                             'test': torch.utils.data.DataLoader(image_datasets['test'], 
#                                             batch_sampler=test_batch_sampler, num_workers=1),
#                             }

#         else:
#             sampler = {x: None for x in ['train', 'test']}
#             dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
#                                             shuffle=True, num_workers=1)
#                                             for x in ['train', 'test']}

#     elif contrastive == True:
#         color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
#         data_transforms = {'augmentation': transforms.Compose([transforms.RandomResizedCrop(size=img_size),
#                                               transforms.RandomHorizontalFlip(),
#                                             #   transforms.RandomApply([color_jitter], p=0.5),
#                                               transforms.RandomGrayscale(p=1),
#                                             #   data_aug.GaussianBlur(kernel_size=int(0.1 * img_size + 1)),
#                                               transforms.ToTensor()]),
                            
#                             'train': transforms.Compose([
#                                         transforms.Pad(4),
#                                         transforms.RandomCrop(96),
#                                         transforms.Resize(img_size),
#                                         transforms.RandomHorizontalFlip(),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(mean, std),
#                                 ]),
#                             }
#         image_datasets = {x: torchvision.datasets.STL10(root=os.path.join(data_dir, x), split=x,
#                                                 transform=SimCLRDataTransform(data_transforms), download=True)
#                             for x in ['train', 'test']}
#         dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
#                                         shuffle=True, num_workers=1)
#                 for x in ['train', 'test']}

#     return image_datasets, dataloaders

    
def get_std_and_mean(dataset):
    if dataset == 'stl10':
        print ("Get the std and mean for stl10 dataset...")
        mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
    else:
        print ("Get the std and mean for imagenet series dataset...")
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    return mean, std


def recover_image(images, std, mean):
    images[:, 0, :, :] = images[:, 0, :, :] * std[0] + mean[0]
    images[:, 1, :, :] = images[:, 1, :, :] * std[1] + mean[1]
    images[:, 2, :, :] = images[:, 2, :, :] * std[2] + mean[2]

    return images 

def create_val_img_folder(args):
    '''
    This method is responsible for separating validation images into separate sub folders (imagenet test folder need this method)
    '''
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


if __name__ == '__main__':
    # create val img folder for tinyimagenet_data
    # data_dir = '/research/dept2/yuli/datasets/imagenet'
    # dataset = 'tinyimagenet_data'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='data directory')
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    args = parser.parse_args()

    create_val_img_folder(args)

