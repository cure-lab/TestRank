#!/bin/bash

############### Function Declarations   ##############################
# The input is to choose a function to run: train_model or test_selection
    # train_model: train several biased model using train set and val set for IP vendor
    # test_sel: test the model from IP vendor and generate results
# History
    # Author: YU LI, Email: yu.li.sallylee@gmail.com; Last modified: May 15, 2020

# echo -e "please input the function to execute, select from \
#             (train_model: trainm,  
#              test_selection: selection_v2,
# read function

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

DATE=`date +%Y-%m-%d`
echo $DATE
DIRECTORY=./save/${DATE}/
if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Step selection   ##############################

function="$1"
echo "Input function: "$function
test -z $function && echo "You must input a function" && exit 0

############### Configuration   ##############################

DATA_ROOT='/research/dept2/yuli/datasets'

epoch=200
STEP=100
RANDOM_SEED=10


############### Train   ##############################
# ----- IP vendor: Train biased models -----
if [ "$function" == "trainm" ]; then
    echo "train model for IP vendor"
    # train models
#    DATASET='emnist'
#    MODEL='resnet34'
    DATASET='stl10'
    MODEL='resnet34'
    save_path=save/${DATE}/${DATASET}_${MODEL}

    python train_classifier.py --dataset ${DATASET} \
                                --model ${MODEL} \
                                --n_epochs ${epoch} \
                                --data_root ${DATA_ROOT} \
                                --manualSeed ${RANDOM_SEED} \
                                --save_path ${save_path} \
                                --class_weight 0 

    wait 

    python train_classifier.py --dataset ${DATASET} \
                                --model ${MODEL} \
                                --n_epochs ${epoch} \
                                --data_root ${DATA_ROOT} \
                                --manualSeed ${RANDOM_SEED} \
                                --save_path ${save_path} \
                                --class_weight 1

    wait 

    python train_classifier.py --dataset ${DATASET} \
                                --model ${MODEL} \
                                --n_epochs ${epoch} \
                                --data_root ${DATA_ROOT} \
                                --manualSeed ${RANDOM_SEED} \
                                --save_path ${save_path} \
                                --class_weight 2
    wait
fi 


#  ------Test Center:  Selection ------
DATASET='cifar10'
MODEL='resnet18'

# DATASET='svhn'
# MODEL='wide_resnet'

# DATASET='stl10'
# MODEL='resnet34'


####  ------Test Center:  Selection Version 2, fixed mini-budget ------
###### output filename, remove if exists
no_neighbors=100
# 0: BYOL, 1:model2test
feature_extractor_id=0

# run testing
if [ "$function" == "selection_v2" ]; then 
    for MODEL_NO in 0 1 2; do
        echo $MODEL_NO
        MODEL2TEST=${MODEL}
        MODEL2TESTPATH=./checkpoint/${DATASET}/ckpt_bias/${MODEL}_${MODEL_NO}_b.t7 
        save_path=save/${DATE}/${DATASET}_${MODEL2TEST}_${STEP}
        echo 'model to test arch '$MODEL2TEST
        echo 'model to test path '$MODEL2TESTPATH
        SEL_METHOD='random'
        # for no_neighbors in 10 20 50 70 100 150 200 300 400 500 600 700 800; do
            python selection.py \
                    --dataset $DATASET \
                    --manualSeed ${RANDOM_SEED} \
                    --model2test_arch $MODEL2TEST \
                    --model2test_path $MODEL2TESTPATH \
                    --model_number $MODEL_NO \
                    --sel_method $SEL_METHOD \
                    --save_path ${save_path} \
                    --data_path ${DATA_ROOT} \
                    --graph_nn \
                    --feature_extractor_id ${feature_extractor_id} \
                    --no_neighbors ${no_neighbors} \
                    --learn_mixed 
                    # --latent_space_plot
        # done
    done

fi


