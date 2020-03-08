# create toy dataset, along with task selection
# manage toy dataset during train/test
# author: satwik kottur

import torch
import functools
import itertools, pdb, json, random

class Dataloader:

    # initialize
    def __init__(self, params):
        # absorb all values from params
        for field, value in params.items():
          setattr(self, field, value)

        # if loadPath is given, load dataset
        if 'dataset' not in params:
            print('Creating empty dataloader!')
            return

        self.loadDataset(params['dataset'])
        ####################### Create attributes #########################
        numVals = {attr:len(vals) for attr, vals in self.props.items()}
        self.attrValVocab = functools.reduce(lambda x, y: x + y,
            [self.props[ii] for ii in self.attributes])
        self.numTasks = len(self.taskDefn)

        # input vocab for answerer
        # inVocab and outVocab same for questioner
        taskVocab = ['<T%d>' % ii for ii in range(self.numTasks)]
        # A, Q have different vocabs
        qOutVocab = [chr(ii + 97) for ii in range(params['qOutVocab'])]
        aOutVocab = [chr(ii + 65) for ii in range(params['aOutVocab'])]
        aInVocab =  qOutVocab + aOutVocab
        qInVocab = aOutVocab + qOutVocab + taskVocab

        # pack parameters
        self.params = {'numTasks': self.numTasks, 'taskSelect': self.taskDefn,\
                       'props': self.props, 'attributes': self.attributes,\
                       'qOutVocab':len(qOutVocab), 'qInVocab':len(qInVocab),\
                       'aOutVocab':len(aOutVocab), 'aInVocab':len(aInVocab)}

        self.numAttrs = len(self.attributes)
        self.taskSelect = torch.LongTensor(self.taskDefn)

        # number of single and pair wise tasks
        self.numPairTasks = 6
        self.numSingleTasks = 3

        # create a vocab map for field values
        attrVals = functools.reduce(lambda x, y: x+y,
                                    [self.props[ii] for ii in self.attributes])
        self.attrVocab = {value: ii for ii, value in enumerate(attrVals)}
        self.invAttrVocab = {index: attr for attr, index in self.attrVocab.items()}

        # get encoding for attribute pairs
        self.attrPair = itertools.product(attrVals, repeat=2)
        self.attrPairVocab = {value:ii for ii, value in enumerate(self.attrPair)}
        self.invAttrPairVocab = {index:value for value, index \
                                                in self.attrPairVocab.items()}

        # Separate data loading for test/train
        self.data = {}
        for dtype in ['train', 'test']:
            data = torch.LongTensor(self.numInst[dtype], self.numAttrs)
            for ii, attrSet in enumerate(self.split[dtype]):
                data[ii] = torch.LongTensor([self.attrVocab[at] for at in attrSet])
            self.data[dtype] = data

        self.rangeInds = torch.arange(0, self.numInst['train']).long()
        # ship to gpu if needed
        if self.useGPU:
            for key, value in self.data.items():
                self.data[key] = value.cuda()
            self.rangeInds = self.rangeInds.cuda()

    # load dataset
    def loadDataset(self, loadPath):
        # load and absorb the values
        with open(loadPath, 'r') as fileId: loaded = json.load(fileId)
        for key, value in loaded.items(): setattr(self, key, value)

    # create and save the dataset
    def saveDataset(self, savePath, trainSize=0.8):
        attributes = ['colors', 'shapes', 'styles']
        # larger dataset
        #props = {'colors': ['red', 'green', 'blue', 'purple', \
        #                    'yellow', 'cyan', 'orange', 'teal'], \
        #        'shapes': ['square', 'triangle', 'circle', 'star', \
        #                    'heart', 'pentagon', 'hexagon', 'ring'],\
        #       'styles': ['dotted', 'solid', 'filled', 'dashed', 'hstripe', \
        #                   'vstripe', 'hgradient', 'vgradient']}
        props = {'colors': ['red', 'green', 'blue', 'purple'],\
                'shapes': ['square', 'triangle', 'circle', 'star'], \
                'styles': ['dotted', 'solid', 'filled', 'dashed']}
        attrList = [props[ii] for ii in attributes]
        dataVerbose = list(itertools.product(*attrList))

        # select trainSize for train
        numImgs = len(dataVerbose)
        numInst = {}
        numInst['train'] = int(trainSize * numImgs)
        numInst['test'] = numImgs - numInst['train']

        # randomly select test
        splitData = {}
        splitData['test'] = random.sample(dataVerbose, numInst['test'])
        splitData['train'] = list(set(dataVerbose) - set(splitData['test']))

        # six tasks, including the order
        taskDefn = [[0, 1], [1, 0], [0, 2], \
                    [2, 0], [1, 2], [2, 1], \
                    [0, 0], [1, 1], [2, 2]]

        toSave = {'attributes':attributes, 'props':props, 'taskDefn':taskDefn,\
                    'numInst':numInst, 'split':splitData}

        # perform sanity check to make sure every attribute in test is seen
        attrListTest = set([jj for ii in splitData['test'] for jj in ii])
        attrListTrain = set([jj for ii in splitData['train'] for jj in ii])
        assert attrListTest.issubset(attrListTest), 'Test has unknown attributes'

        print(numInst)
        print('Saving dataset: ' + savePath)
        with open(savePath, 'w') as fileId: json.dump(toSave, fileId)

    #  query number of instances
    def getInstCount(self): return self.numInst

    # get a batch
    def getBatch(self, batchSize):
        # sample tasks
        tasks = torch.LongTensor(batchSize).random_(0, self.numPairTasks)
        # sample a batch
        indices = torch.LongTensor(batchSize).random_(0, self.numInst['train'])
        if self.useGPU: indices = indices.cuda()
        batch = self.data['train'][indices]

        # now sample predictions based on task
        selectInds = self.taskSelect[tasks]
        if self.useGPU:
            selectInds = selectInds.cuda()
            tasks = tasks.cuda()
        labels = batch.gather(1, selectInds)

        return batch, tasks, labels

    # get a batch
    def getBatchSpecial(self, batchSize, currentPred, negFraction=0.8):
        # sample tasks
        tasks = torch.LongTensor(batchSize).random_(0, self.numPairTasks)
        # sample a batch
        indices = torch.LongTensor(batchSize).random_(0, self.numInst['train'])
        if self.useGPU: indices = indices.cuda()
        #-------------------------------------------------------------
        # fill the first batchSize/2 based on previously misclassified examples
        negInds = currentPred.view(-1, self.numPairTasks).sum(1) < self.numPairTasks
        negInds = self.rangeInds.masked_select(negInds)
        negBatchSize = int(batchSize * negFraction)
        # sample from this
        negSamples = torch.LongTensor(negBatchSize).fill_(0)
        if negInds.size(0) > 1: negSamples.random_(0, negInds.size(0))
        if self.useGPU: negSamples = negSamples.cuda()
        negInds = negInds[negSamples]
        indices[:negBatchSize] = negInds
        #-------------------------------------------------------------
        batch = self.data['train'][indices]

        # now sample predictions based on task
        selectInds = self.taskSelect[tasks]
        if self.useGPU:
            selectInds = selectInds.cuda()
            tasks = tasks.cuda()
        labels = batch.gather(1, selectInds)

        return batch, tasks, labels

    # Get all configurations
    def getCompleteData(self, dtype):
        # expand self.data three folds, along with labels
        batch = self.data[dtype].unsqueeze(0).repeat(1, 1, self.numPairTasks)
        batch = batch.view(-1, self.numAttrs)
        tasks = torch.arange(0, self.numPairTasks).long()
        tasks = tasks.unsqueeze(0).repeat(1, self.numInst[dtype]).view(-1)

        # now sample predictions based on task
        selectInds = self.taskSelect[tasks]
        if self.useGPU:
            selectInds = selectInds.cuda()
            tasks = tasks.cuda()
        labels = batch.gather(1, selectInds)
        return batch, tasks, labels

    # converting to text
    def reformatTalk(self, talk, preds, images, tasks, labels):
        script = []
        numImgs = images.size(0)
        if self.qOutVocab < 4:
            aVocab = [str(ii) for ii in range(self.aOutVocab)]
            qVocab = [chr(ii + 88) for ii in range(self.qOutVocab)]
        else:
            aVocab = ['a-%d' % ii for ii in range(self.aOutVocab)]
            qVocab = ['q-%d' % ii for ii in range(self.qOutVocab)]

        attrPairInv = {ii:value for value, ii in self.attrPairVocab.items()}
        for ii in range(numImgs):
            # conversation
            conv = {}
            conv['image'] = [self.invAttrVocab[jj.item()] for jj in images[ii]]
            conv['gt'] = [self.invAttrVocab[labels[ii, jj].item()]
                          for jj in range(2)]
            conv['task'] = [self.attributes[jj.item()]
                                        for jj in self.taskSelect[tasks[ii]]]
            conv['pred'] = [self.invAttrVocab[preds[jj].data[ii].item()]
                                                for jj in range(2)]
            conv['chat'] = [qVocab[talk[0].data[ii]],
                            aVocab[talk[1].data[ii]]]
            if len(talk) > 3:
                conv['chat'].extend([qVocab[talk[2].data[ii]],
                                    aVocab[talk[3].data[ii]]])
            script.append(conv)

        #self.prettyPrint(script)
        # re-arrange such that negative examples are on the top
        wrongEx = []
        for ii in script:
            if ii['gt'] != ii['pred']: wrongEx.append(ii)

        # remove wrong Ex from script
        for ex in wrongEx: script.remove(ex)
        # append both
        script = wrongEx + script
        return script

    # Pretty print result
    def prettyPrint(self, talk):
        for conv in talk:
            # first print image, task
            print('Im: %s -  Task: %s' % (conv['image'], conv['task']))
            # print conversation
            print('\tQ1 : %s \t A1: %s' % (conv['chat'][0], conv['chat'][1]))
            print('\tQ2 : %s \t A2: %s' % (conv['chat'][2], conv['chat'][3]))
            # print GT and prediction
            print('\tGT: %s\tPred: %s' % (conv['gt'], conv['pred']))
            print('--------------------\n')

###############################################################################
# main to dump the dataset
if __name__ == '__main__':
    options = {}
    # create dataloader
    data = Dataloader(options)
    data.saveDataset('data/toy64_split_0.8.json', 0.8)
