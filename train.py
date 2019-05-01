# script to train interactive bots in toy world
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools, pdb, random, pickle, os
import numpy as np
from chatbots import Team
from dataloader import Dataloader
import options
from time import gmtime, strftime
#import matplotlib.pyplot as plt

# read the command line options
options = options.read();
#------------------------------------------------------------------------
# setup experiment and dataset
#------------------------------------------------------------------------
data = Dataloader(options);
numInst = data.getInstCount();

params = data.params;
# append options from options to params
for key, value in options.iteritems(): params[key] = value;

#------------------------------------------------------------------------
# build agents, and setup optmizer
#------------------------------------------------------------------------
team = Team(params);
team.train();
optimizer = optim.Adam([{'params': team.aBot1.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.qBot1.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.aBot2.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.qBot2.parameters(), \
                                'lr':params['learningRate']}])
#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
# begin training
numIterPerEpoch = int(np.ceil(numInst['train']/params['batchSize']));
numIterPerEpoch = max(1, numIterPerEpoch);
count = 0;
savePath = 'models/tasks_inter_%dH_%.4flr_%r_%d_%d.pickle' %\
            (params['hiddenSize'], params['learningRate'], params['remember'],\
            options['aOutVocab'], options['qOutVocab']);

matches1 = {};
accuracy1 = {};
matches2 = {};
accuracy2 = {};
trainAccHistory1 = [];
testAccHistory1 = [];
trainAccHistory2 = [];
testAccHistory2 = [];
for iterId in xrange(params['numEpochs'] * numIterPerEpoch):
    epoch = float(iterId)/numIterPerEpoch;

    # get double attribute tasks
    if 'train' not in matches1:
        batchImg1, batchTask1, batchLabels1 \
                            = data.getBatch(params['batchSize']);
    else:
        batchImg1, batchTask1, batchLabels1 \
                = data.getBatchSpecial(params['batchSize'], matches1['train'],\
                                                        params['negFraction']);
    if 'train' not in matches2:
        batchImg2, batchTask2, batchLabels2 \
                            = data.getBatch(params['batchSize']);
    else:
        batchImg2, batchTask2, batchLabels2 \
                = data.getBatchSpecial(params['batchSize'], matches2['train'],\
                                                        params['negFraction']);

    # forward pass
    team.forward(Variable(batchImg1), Variable(batchTask1), Variable(batchImg2),\
        Variable(batchTask2));
    # backward pass
    team.backward(optimizer, batchLabels1, batchLabels2, epoch);

    # take a step by optimizer
    optimizer.step()
    #--------------------------------------------------------------------------
    # switch to evaluate
    team.evaluate();

    for dtype in ['train', 'test']:
        # get the entire batch
        img, task, labels = data.getCompleteData(dtype);
        # evaluate on the train dataset, using greedy policy
        guess1,guess2,_,_,_,_ = team.forward(Variable(img), Variable(task),\
            Variable(img), Variable(task));
        # compute accuracy for color, shape, and both
        firstMatch1 = guess1[0].data == labels[:, 0].long();
        secondMatch1 = guess1[1].data == labels[:, 1].long();
        matches1[dtype] = firstMatch1 & secondMatch1;
        accuracy1[dtype] = 100*torch.sum(matches1[dtype]).float()\
                                    /float(matches1[dtype].size(0));
        firstMatch2 = guess2[0].data == labels[:, 0].long();
        secondMatch2 = guess2[1].data == labels[:, 1].long();
        matches2[dtype] = firstMatch2 & secondMatch2;
        accuracy2[dtype] = 100*torch.sum(matches2[dtype]).float()\
                                    /float(matches2[dtype].size(0));
    # switch to train
    team.train();

    # break if train accuracy reaches 100%
    if accuracy1['train'] == 100 or accuracy2['train'] == 100: break;

    # save for every 5k epochs
    if iterId > 0 and iterId % (10000*numIterPerEpoch) == 0:
        team.saveModel(savePath, optimizer, params);
        historySavePath = savePath.replace('inter', 'history')
        with open(historySavePath, 'wb') as f:
            pickle.dump({
                    'train1': trainAccHistory1,
                    'test1': testAccHistory1,
                    'train2': trainAccHistory2,
                    'test2': testAccHistory2
                }, f)

    if iterId % 100 != 0: continue;

    time = strftime("%a, %d %b %Y %X", gmtime());
    print('[%s][Iter: %d][Ep: %.2f][R1: %.4f][Tr1: %.2f Te1: %.2f]' % \
                                (time, iterId, epoch, team.totalReward1,\
                                accuracy1['train'], accuracy1['test']))
    print('[%s][Iter: %d][Ep: %.2f][R2: %.4f][Tr2: %.2f Te2: %.2f]' % \
                                (time, iterId, epoch, team.totalReward2,\
                                accuracy2['train'], accuracy2['test']))
    trainAccHistory1.append(accuracy1['train']);
    testAccHistory1.append(accuracy1['test']);
    trainAccHistory2.append(accuracy2['train']);
    testAccHistory2.append(accuracy2['test']);
#------------------------------------------------------------------------
# save final model with a time stamp
timeStamp = strftime("%a-%d-%b-%Y-%X", gmtime());
replaceWith = 'final_%s' % timeStamp;
finalSavePath = savePath.replace('inter', replaceWith);
print('Saving : ' + finalSavePath)
team.saveModel(finalSavePath, optimizer, params);
#------------------------------------------------------------------------
# Plot train and test accuracy over epochs
#plt.plot(trainAccHistory1);
#plt.plot(testAccHistory1);
#plt.plot(trainAccHistory2);
#plt.plot(testAccHistory2);
#plt.title('Accuracy vs Epochs');
#plt.xlabel('Epochs (x100)');
#plt.ylabel('Accuracy (%)');
#plt.show();
historySavePath = finalSavePath.replace('final', 'history')
with open(historySavePath, 'wb') as f:
    pickle.dump({
            'train1': trainAccHistory1,
            'test1': testAccHistory1,
            'train2': trainAccHistory2,
            'test2': testAccHistory2
        }, f)
