# TestRank in Pytorch

Code for the paper [TestRank: Bringing Order into Unlabeled Test Instances for Deep Learning Tasks](https://arxiv.org/abs/2105.10113) by Yu Li, Min Li, Qiuxia Lai, Yannan Liu, and Qiang Xu. 

If you use this code, or development from it, please cite our paper:
```
@article{yu2021testrank,
  title={TestRank: Bringing Order into Unlabeled Test Instances for Deep Learning Tasks},
  author={Yu Li, Min Li, Qiuxia Lai, Yannan Liu, and Qiang Xu},
  journal={NeurIPS},
  year={2021}
}
```


## 1. Setup

Install dependencies
```setup
conda env create -f environment.yml
```

Please run the code on GPU.


## 2. Runing
<!-- In general, we have the following main files:

      --train_classifier.py # used to train DL classifiers
      --selection.py  # the selection strategy
      --byol/* # code to train the unsupervised feature extractor -->

There are mainly **three** steps involved:
   - Prepare the DL models to be tested
   - Prepare the unsupervised BYOL feature extractor 
   - Launch a specific test input prioritization technique

We illustrate these steps as the following.
 
### 2.1. Download the Pre-trained DL model under test
 
 - [ResNet-18](https://github.com/yuli-yl/TestRank/tree/master/checkpoint/cifar10/ckpt_bias) trained on CIFAR10 dataset. 

- [Wide-ResNet](https://github.com/yuli-yl/TestRank/tree/master/checkpoint/svhn/ckpt_bias) trained on SVHN dataset.

- [ResNet-34](https://github.com/yuli-yl/TestRank/tree/master/checkpoint/stl10/ckpt_bias) trained on STL10 dataset.

Please download the classifiers to corresponding folder  **./checkpoint/{dataset}/ckpt_bias/**

  <!-- To make sure the correctness of the downloaded file, please check your **md5** after you download the weight files:

      --resnet34_0_b.t7: e6e518998e9be957c77afe8a33aff590
      --resnet34_1_b.t7: 44a5f49cc833421f0e489a5e0aa37bac
      --resnet34_2_b.t7: 388598538a54aa2f96c082c07a08fbc3 -->


If you want to train your own classifiers, please refer to the **Training** part.


### 2.2. Download the Feature extractor

We papare pretrained [feature extractor](https://github.com/yuli-yl/TestRank/tree/master/ckpt_byol) for the each (e.g. CIFAR-10, SVHN, STL10) dataset. Please put the downloaded file in the "./ckpt_byol/" folder.

<!-- The md5 of this file is: fe7e3bc9f846e0250c7e6951034ec13f -->
If you want to train your own classifiers, please refer to the **Training** part.

### 2.3. Perform Test Selection
   Call the 'run.sh' file with argument 'selection':

      ./run.sh selection

   Configure your run.sh follow the discription below
      
      python selection.py \
                  --dataset $DATASET \                   # specify the dataset to use
                  --manualSeed ${RANDOM_SEED} \          # random seed
                  --model2test_arch $MODEL2TEST \        # architecture of the model under test (e.g. resnet18)
                  --model2test_path $MODEL2TESTPATH \    # the path storing the model weights 
                  --model_number $MODEL_NO \             # which model to test, model 0, 1, or 2?
                  --save_path ${save_path} \             # The result will be stored in here
                  --data_path ${DATA_ROOT} \             # Dataset root path
                  --graph_nn \                           # use graph neural network in testrank
                  --feature_extractor_id ${feature_extractor_id} \ # type of feature extractor, 0: BYOL model, 1: the model under test
                  --no_neighbors ${no_neighbors} \       # number of neighbors in to constract graph
                  --learn_mixed                          # use mlp to combine intrinsic and contextual attributes; otherwise they are brute force combined (multiplication two scores)
                  --baseline_gini                        # Use certain baseline method to perform selection, otherwise leave it blank

   - The result is stored in '{save_path}/{date}/{dataset}_{model}/xxx_result.csv' in where xxx stands for the selection method used (e.g. for testrank, the file would be gnn_result.csv)

   
   - The **TRC** value is in the last column, and the forth column shows the corresponding budget in percent.

   - To compare with baselines, please specify the corresponding baseline method (e.g. baseline_gini, baseline_uncertainty, baseline_dsa, baseline_mcp):

   - To evaluate different models, change the MODEL_NO to the corresponding model: [0, 1, 2]

## 3. Training

### 3.1. Train classifier
If you want to train your own DL model instead of using the pretrained ones, run this command:

```train
./run.sh trainm
```

- The trained model will be stored in path './checkpoint/dataset/ckpt_bias/*'. 

 - Each model will be assigned with a unique ID (e.g. 0, 1, 2). 
   

- The code used to train the model are resides in the **train_classifier.py** file. 
If you want to change the dataset or model architecture, please modify 'DATASET=dataset_name' or 'MODEL=name'with the desired ones in the **run.sh** file.

### 3.2 Train BYOL Feature Extractor
Please refer to this [code](https://github.com/sthalles/PyTorch-BYOL).


## 4. Contact
If there are any questions, feel free to send a message to yuli@cse.cuhk.edu.hk


