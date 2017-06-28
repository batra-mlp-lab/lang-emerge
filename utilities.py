# methods to help
# author: satwik kottur

import torch
import sys, json, pdb, math
sys.path.append('../')
from html import HTML

# Initializing weights
def initializeWeights(moduleList, itype):
    assert itype=='xavier', 'Only Xavier initialization supported';

    for moduleId, module in enumerate(moduleList):
        if hasattr(module, '_modules') and len(module._modules) > 0:
            # Iterate again
            initializeWeights(module, itype);
        else:
            # Initialize weights
            name = type(module).__name__;
            # If linear or embedding
            if name == 'Embedding' or name == 'Linear':
                fanIn = module.weight.data.size(0);
                fanOut = module.weight.data.size(1);

                factor = math.sqrt(2.0/(fanIn + fanOut));
                weight = torch.randn(fanIn, fanOut) * factor;
                module.weight.data.copy_(weight);

            # If LSTMCell
            if name == 'LSTMCell':
                for name, param in module._parameters.iteritems():
                    if 'bias' in name:
                        module._parameters[name].data.fill_(0.0);
                        #print('Initialized: %s' % name)

                    else:
                        fanIn = param.size(0);
                        fanOut = param.size(1);

                        factor = math.sqrt(2.0/(fanIn + fanOut));
                        weight = torch.randn(fanIn, fanOut) * factor;
                        module._parameters[name].data.copy_(weight);
                        #print('Initialized: %s' % name)

            # Check for bias and reset
            if hasattr(module, 'bias') and type(module.bias) != bool:
                module.bias.data.fill_(0.0);

def saveResultPage(loadPath):
    # image, task, converation, GT, pred
    page = HTML(5);
    page.setTitle(['Image', 'Task', 'Conversation', 'GT', 'Pred']);

    savePath = loadPath.replace('json', 'html').replace('chatlog', 'chatpage');

    with open(loadPath, 'r') as fileId: talk = json.load(fileId);

    maps = {'rectangle':'triangle', 'rhombus':'star', 'cyan':'purple'};
            #'A':'  I', 'B':' II', 'C':'III'};
    cleaner = lambda x: maps[x] if x in maps else x;

    for datum in talk:
        datum['image'] = [cleaner(ii) for ii in datum['image']];
        datum['gt'] = [cleaner(ii) for ii in datum['gt']];
        datum['pred'] = [cleaner(ii) for ii in datum['pred']];
        datum['chat'] = [cleaner(ii) for ii in datum['chat']];

        row = [', '.join(datum['image']), ', '.join(datum['task'])];

        # add chat
        chat = 'Q1 : %3s \tA1: %s ' % (datum['chat'][0], datum['chat'][1]);
        if len(datum['chat']) > 3:
            chat += '\tQ2 : %3s \t A2: %s' % (datum['chat'][2], datum['chat'][3]);
        row.append(chat)

        # add GT and pred
        row.extend([', '.join(datum['gt']), ', '.join(datum['pred'])]);

        page.addRow(row);

    # render and save page
    page.savePage(savePath);

def saveResultPagePool(loadPath):
    # image, task, converation, GT, pred
    page = HTML(4);
    page.setTitle(['Pool', 'GT', 'Conversation', 'Pred']);

    savePath = loadPath.replace('json', 'html').replace('chatlog', 'chatpage');

    with open(loadPath, 'r') as fileId: talk = json.load(fileId);

    maps = {};
    cleaner = lambda x: maps[x] if x in maps else x;

    for datum in talk:
        datum['pool'] = [[cleaner(jj) for jj in ii] \
                                        for ii in datum['pool']];
        datum['gt'] = [cleaner(ii) for ii in datum['gt']];
        datum['pred'] = [cleaner(ii) for ii in datum['pred']];
        datum['chat'] = [cleaner(ii) for ii in datum['chat']];

        row = ['\n'.join([', '.join(ii) for ii in datum['pool']])];
        row.append(', '.join(datum['gt']));

        # add chat
        chat = 'Q1 : %3s \tA1: %s ' % (datum['chat'][0], datum['chat'][1]);
        if len(datum['chat']) > 3:
            chat += '\tQ2 : %3s \t A2: %s' % (datum['chat'][2], datum['chat'][3]);
        row.append(chat)

        # add GT and pred
        row.append(', '.join(datum['pred']));
        page.addRow(row);

    # render and save page
    page.savePage(savePath);
