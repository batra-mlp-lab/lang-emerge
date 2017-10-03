DATA_PATH='data/toy64_split_0.8.json'
VISIBLE_CUDA_DEVICES=0
# script to run the program
python train.py -learningRate 0.01 -hiddenSize 100 -batchSize 1000 \
                -imgFeatSize 20 -embedSize 20\
                -useGPU -dataset $DATA_PATH\
                -aOutVocab 4 -qOutVocab 3
