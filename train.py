# script to train interactive bots in toy world
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools, pdb, random, os
import numpy as np
from chatbots import Team
from dataloader import Dataloader
import options
from time import gmtime, strftime

# read the command line options
options = options.read()
#------------------------------------------------------------------------
# setup experiment and dataset
#------------------------------------------------------------------------
data = Dataloader(options)
numInst = data.getInstCount()

params = data.params
# append options from options to params
for key, value in options.items():
  params[key] = value

#------------------------------------------------------------------------
# build agents, and setup optmizer
#------------------------------------------------------------------------
team = Team(params)
team.train()
optimizer = optim.Adam([{'params': team.aBot.parameters(), \
                                'lr':params['learningRate']},\
                        {'params': team.qBot.parameters(), \
                                'lr':params['learningRate']}])
#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
# begin training
numIterPerEpoch = int(np.ceil(numInst['train']/params['batchSize']))
numIterPerEpoch = max(1, numIterPerEpoch)
count = 0
savePath = 'models/tasks_inter_%dH_%.4flr_%r_%d_%d.tar' %\
            (params['hiddenSize'], params['learningRate'], params['remember'],\
            options['aOutVocab'], options['qOutVocab'])

matches = {}
accuracy = {}
bestAccuracy = 0
for iterId in range(params['numEpochs'] * numIterPerEpoch):
    epoch = float(iterId)/numIterPerEpoch

    # get double attribute tasks
    if 'train' not in matches:
        batchImg, batchTask, batchLabels \
                            = data.getBatch(params['batchSize'])
    else:
        batchImg, batchTask, batchLabels \
                = data.getBatchSpecial(params['batchSize'], matches['train'],\
                                                        params['negFraction'])

    # forward pass
    team.forward(Variable(batchImg), Variable(batchTask))
    # backward pass
    batchReward = team.backward(optimizer, batchLabels, epoch)

    # take a step by optimizer
    optimizer.step()
    #--------------------------------------------------------------------------
    # switch to evaluate
    team.evaluate()

    for dtype in ['train', 'test']:
        # get the entire batch
        img, task, labels = data.getCompleteData(dtype)
        # evaluate on the train dataset, using greedy policy
        guess, _, _ = team.forward(Variable(img), Variable(task))
        # compute accuracy for color, shape, and both
        firstMatch = guess[0].data == labels[:, 0].long()
        secondMatch = guess[1].data == labels[:, 1].long()
        matches[dtype] = firstMatch & secondMatch
        accuracy[dtype] = 100*torch.sum(matches[dtype])\
                                    /float(matches[dtype].size(0))
    # switch to train
    team.train()

    # break if train accuracy reaches 100%
    if accuracy['train'] == 100: break

    # save for every 5k epochs
    # if iterId > 0 and iterId % (10000*numIterPerEpoch) == 0:
    if iterId >= 0 and iterId % (10000*numIterPerEpoch) == 0:
        team.saveModel(savePath, optimizer, params)

    if iterId % 100 != 0: continue

    time = strftime("%a, %d %b %Y %X", gmtime())
    print('[%s][Iter: %d][Ep: %.2f][R: %.4f][Tr: %.2f Te: %.2f]' % \
                                (time, iterId, epoch, team.totalReward,\
                                accuracy['train'], accuracy['test']))
#------------------------------------------------------------------------
# save final model with a time stamp
timeStamp = strftime("%a-%d-%b-%Y-%X", gmtime())
replaceWith = 'final_%s' % timeStamp
finalSavePath = savePath.replace('inter', replaceWith)
print('Saving : ' + finalSavePath)
team.saveModel(finalSavePath, optimizer, params)
#------------------------------------------------------------------------
