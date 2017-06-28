# script to learn the language chart
# author: satwik kottur

import json, os, sys, pdb, subprocess
from tqdm import tqdm as progressbar
from collections import defaultdict

# segregrate json files
def separateJSON(listing):
    jsonFiles = [fileName for fileName in listing if 'json' in fileName];

    # arrange based on the epoch
    if len(jsonFiles) > 0:
        getEpoch = lambda x: int(x.strip('.json').split('_')[-1])
        sortedOrder = sorted(jsonFiles, key=getEpoch);
        return sortedOrder;
    else: return False;

# create json
def createJSON(folderPath, listing):
    print('Creating JSON files: ' + folderPath);

    commandFmt = 'python test.py %s';
    for fileName in listing:
        subprocess.call(commandFmt % (folderPath + fileName), shell=True);

# compute accuracy
def computeAccuracy(jsonPath):
    with open(jsonPath, 'r') as fileId: data = json.load(fileId);
    # number of instances
    numInst = len(data);
    # number of correct
    numCorrect = len([ii for ii in data if ii['pred'] == ii['gt']]);
    accuracy = (100 * float(numCorrect)/numInst);
    return accuracy;
    #print('Accuracy: %f' % (100 * float(numCorrect)/numInst))

# building the dialog tree
def buildDialogTree(jsonPath):
    with open(jsonPath, 'r') as fileId: data = json.load(fileId);

    # figure out the vocab sizes
    qChat = [item for ii in data for item in ii['chat'][::2]];
    aChat = [item for ii in data for item in ii['chat'][1::2]];
    qVocab = len(set(qChat));
    aVocab = len(set(aChat));

    # dictionary of conversations and (image, task)
    tree = defaultdict(set);
    for datum in data:
        imageTask = tuple(datum['image'] + datum['task']);
        chat = datum['chat'];

        # add four levels
        tree[tuple(chat[0:1])].add(imageTask);
        tree[tuple(chat[0:2])].add(imageTask);
        tree[tuple(chat[0:3])].add(imageTask);
        tree[tuple(chat)].add(imageTask);

    return tree;

# check for language chart trends
def obtainLanguageChart(forest):
    trends = [];
    for pos in forest[-1].keys():
        current = forest[0][pos];
        currentId = 0;
        for treeId, tree in enumerate(forest):
            #if tree[pos] != current:
            # one has to be a subset of another
            if not tree[pos].issubset(current) and not current.issubset(tree[pos]):
                currentId = treeId;
                current = tree[pos];

        # print if there is a trend
        if currentId != len(forest):
            if len(forest[currentId][pos]) == 0: continue;
            if len(forest[-1][pos]) == 0: continue;

            # Check for attributes common among the nodes
            finalMembers = forest[-1][pos];
            trend = set.intersection(*[set(ii) for ii in finalMembers]);
            trends.append((currentId, pos, trend))

    trends = sorted(trends, key=lambda x:x[0]);
    for trend in trends:
        print('\nTrend found [%d / %d]:' % (trend[0], len(forest))),
        print(trend[1])
        print(trend[2])

# check for language chart trends
def backtrackLanguageChart(forest):
    trends = [];
    tasks = {};
    for pos in forest[-1].keys():
        final = forest[-1][pos];
        if len(final) == 0: continue;
        trend = set.intersection(*[set(ii) for ii in final]);
        #print(trend, pos)
        # Go backward in time and check for purity
        learntEpoch = None;
        startEpoch = 0;
        for treeId, tree in enumerate(forest[::-1]):
            # find the start epoch
            # node should be pure
            if len(tree[pos]) == 0:
                if learntEpoch is None: learntEpoch = len(forest) - treeId;
                startEpoch = len(forest) - treeId;
                break;
            else: curTrend = set.intersection(*[set(ii) for ii in tree[pos]]);
            if curTrend != trend:
                if learntEpoch is None: learntEpoch = len(forest) - treeId;

            # figure out number of members following trend
            # go back until at least one of the member has this trend
            if learntEpoch == None: continue;
            members = [trend.issubset(set(ii)) for ii in tree[pos]];
            if sum(members) == 0:
                startEpoch = len(forest) - treeId;
                break;

        trends.append((learntEpoch, startEpoch, pos, list(trend)));
    trends = sorted(trends, key=lambda x:x[0]);

    # save as json file
    savePath = 'trend-dump.json';
    print('Saving trends: ' + savePath)
    with open(savePath, 'w') as fileId: json.dump(trends, fileId);
    for trend in trends:
        # print only 2 or 3 attribute
        if len(trend[2]) > 3: continue;
        try: print('[%d - %d / %d]:' % (trend[1], trend[0], len(forest))),
        except: pdb.set_trace();
        print(trend[2]),
        print(trend[3])
#-------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Wrong usage!')
        print('python <folder>')
        sys.exit(0);

    folderPath = sys.argv[1];
    print('Searching folder: ' + folderPath);
    folderPath = folderPath.strip('/') + '/'; #sanitize
    listing = os.listdir(folderPath);

    jsonFiles = separateJSON(listing);
    # Create json files 
    if not jsonFiles: createJSON(folderPath, listing);
    # read again
    jsonFiles = separateJSON(listing);

    # consider only the train json
    jsonFiles = [folderPath + ii for ii in jsonFiles if 'train' in ii];
    # print accuracies
    #print([computeAccuracy(ii) for ii in jsonFiles]);

    # build the forest
    forest = [buildDialogTree(ii) for ii in jsonFiles];

    # check for consistencies across time
    #obtainLanguageChart(forest);

    # backtrack language chart
    backtrackLanguageChart(forest);
