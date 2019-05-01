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
optimizer = optim.Adam([{'params': team.aBot.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.qBot.parameters(), \
                                'lr':params['learningRate']}]);
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

matches = {};
accuracy = {};
bestAccuracy = 0;
trainAccHistory = [];
testAccHistory = [];
for iterId in xrange(params['numEpochs'] * numIterPerEpoch):
    epoch = float(iterId)/numIterPerEpoch;

    # get double attribute tasks
    if 'train' not in matches:
        batchImg, batchTask, batchLabels \
                            = data.getBatch(params['batchSize']);
    else:
        batchImg, batchTask, batchLabels \
                = data.getBatchSpecial(params['batchSize'], matches['train'],\
                                                        params['negFraction']);

    # forward pass
    team.forward(Variable(batchImg), Variable(batchTask));
    # backward pass
    batchReward = team.backward(optimizer, batchLabels, epoch);

    # take a step by optimizer
    optimizer.step()
    #--------------------------------------------------------------------------
    # switch to evaluate
    team.evaluate();

    for dtype in ['train', 'test']:
        # get the entire batch
        img, task, labels = data.getCompleteData(dtype);
        # evaluate on the train dataset, using greedy policy
        guess, _, _ = team.forward(Variable(img), Variable(task));
        # compute accuracy for color, shape, and both
        firstMatch = guess[0].data == labels[:, 0].long();
        secondMatch = guess[1].data == labels[:, 1].long();
        matches[dtype] = firstMatch & secondMatch;
        accuracy[dtype] = 100*torch.sum(matches[dtype]).float()\
                                    /float(matches[dtype].size(0));
    # switch to train
    team.train();

    # break if train accuracy reaches 100%
    if accuracy['train'] == 100: break;

    # save for every 5k epochs
    if iterId > 0 and iterId % (10000*numIterPerEpoch) == 0:
        team.saveModel(savePath, optimizer, params);
        historySavePath = savePath.replace('inter', 'history')
        with open(historySavePath, 'wb') as f:
            pickle.dump([trainAccHistory, testAccHistory], f)

    if iterId % 100 != 0: continue;

    time = strftime("%a, %d %b %Y %X", gmtime());
    print('[%s][Iter: %d][Ep: %.2f][R: %.4f][Tr: %.2f Te: %.2f]' % \
                                (time, iterId, epoch, team.totalReward,\
                                accuracy['train'], accuracy['test']))
    trainAccHistory.append(accuracy['train']);
    testAccHistory.append(accuracy['test']);
#------------------------------------------------------------------------
# save final model with a time stamp
timeStamp = strftime("%a-%d-%b-%Y-%X", gmtime());
replaceWith = 'final_%s' % timeStamp;
finalSavePath = savePath.replace('inter', replaceWith);
print('Saving : ' + finalSavePath)
team.saveModel(finalSavePath, optimizer, params);
#------------------------------------------------------------------------
# Plot train and test accuracy over epochs
#plt.plot(trainAccHistory);
#plt.plot(testAccHistory);
#plt.title('Accuracy vs Epochs');
#plt.xlabel('Epochs (x100)');
#plt.ylabel('Accuracy (%)');
#plt.show();
historySavePath = finalSavePath.replace('final', 'history')
with open(historySavePath, 'wb') as f:
    pickle.dump([trainAccHistory, testAccHistory], f)
