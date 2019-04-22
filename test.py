# script to develop a toy example
# author: satwik kottur

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools, pdb, random, pickle, json
import numpy as np
from chatbots import Team
from dataloader import Dataloader

import sys
sys.path.append('../');
from utilities import saveResultPage

#------------------------------------------------------------------------
# load experiment and model
#------------------------------------------------------------------------
if len(sys.argv) < 2:
    print('Wrong usage:')
    print('python test.py <modelPath>')
    sys.exit(0);

# load and compute on test
loadPath = sys.argv[1];
print('Loading model from: %s' % loadPath)
with open(loadPath, 'r') as fileId: loaded = pickle.load(fileId);

#------------------------------------------------------------------------
# build dataset, load agents
#------------------------------------------------------------------------
params = loaded['params'];
data = Dataloader(params);

team = Team(params);
team.loadModel(loaded);
team.evaluate();
#------------------------------------------------------------------------
# test agents
#------------------------------------------------------------------------
dtypes = ['train', 'test']
for dtype in dtypes:
    # evaluate on the train dataset, using greedy policy
    images, tasks, labels = data.getCompleteData(dtype);
    # forward pass
    preds1,preds2,_,_,talk1,talk2 = team.forward(Variable(images),\
        Variable(tasks), Variable(images), Variable(tasks), True);

    # compute accuracy for first, second and both attributes
    firstMatch1 = preds1[0].data == labels[:, 0].long();
    secondMatch1 = preds1[1].data == labels[:, 1].long();
    matches1 = firstMatch1 & secondMatch1;
    atleastOne1 = firstMatch1 | secondMatch1;
    firstMatch2 = preds2[0].data == labels[:, 0].long();
    secondMatch2 = preds2[1].data == labels[:, 1].long();
    matches2 = firstMatch2 & secondMatch2;
    atleastOne2 = firstMatch2 | secondMatch2;

    # compute accuracy
    firstAcc1 = 100 * torch.mean(firstMatch1.float());
    secondAcc1 = 100 * torch.mean(secondMatch1.float());
    atleastAcc1 = 100 * torch.mean(atleastOne1.float());
    accuracy1 = 100 * torch.mean(matches1.float());
    firstAcc2 = 100 * torch.mean(firstMatch2.float());
    secondAcc2 = 100 * torch.mean(secondMatch2.float());
    atleastAcc2 = 100 * torch.mean(atleastOne2.float());
    accuracy2 = 100 * torch.mean(matches2.float());
    print('\nTeam 1: Overall accuracy [%s]: %.2f (f: %.2f s: %.2f, atleast: %.2f)'\
                    % (dtype, accuracy1, firstAcc1, secondAcc1, atleastAcc1));
    print('\nTeam 2: Overall accuracy [%s]: %.2f (f: %.2f s: %.2f, atleast: %.2f)'\
                    % (dtype, accuracy2, firstAcc2, secondAcc2, atleastAcc2));

    # pretty print
    talk1 = data.reformatTalk(talk1, preds1, images, tasks, labels);
    talk2 = data.reformatTalk(talk2, preds2, images, tasks, labels);
    if 'final' in loadPath:
        savePath = loadPath.replace('final', 'chatlog-'+dtype);
    elif 'inter' in loadPath:
        savePath = loadPath.replace('inter', 'chatlog-'+dtype);
    savePath1 = savePath.replace('.pickle', '_1.json');
    savePath2 = savePath.replace('.pickle', '_2.json');
    print('Saving conversations: %s' % savePath)
    with open(savePath1, 'w') as fileId:json.dump(talk1, fileId);
    with open(savePath2, 'w') as fileId:json.dump(talk2, fileId);
    saveResultPage(savePath1);
    saveResultPage(savePath2);
