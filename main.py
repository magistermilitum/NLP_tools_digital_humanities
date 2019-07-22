from __future__ import print_function

import os
import sys
import re
import numpy
import pickle
import glob
import argparse
import shutil
import collections
import json

from src.FeatureExtractor import W2VExtractor, HandcraftExtractor
from src.CNNModel import CNNModel
from src.Tokenizer import Tokenizer

BATCH_FORMAT = ".batch"
MODEL_FORMAT = ".json"
WEIGHT_FORMAT = ".h5"

DATA_FORMAT = {
    "word": 0,
    "pos": 1,
    "sct": 2,
    "net": 3
}

NE_LABEL = {
    "O": 0, # 83.25%
    "ORG": 1, # 4.07%
    "PER": 2, # 6.13%
    "LOC": 3, # 4.08%
    "MISC": 4 # 2.47%
}

NE_LOOKUP = ["O", "ORG", "PER", "LOC", "MISC"]

OMITED_DATA = ["\n", "-DOCSTART- -X- O O\n"]

def convert_data(rawDataFile, outputFolder, w2vEx, hcEx, batchSize=None):
    inputFile = open(rawDataFile, "r")
    lines = inputFile.readlines()
    if batchSize is None:
        batchSize = len(lines)
    w2vFeatures = []
    hcFeatures = []
    labels = []
    batch = 0
    for i in range(len(lines)):
        if (len(labels) and len(labels)%batchSize == 0) or i == len(lines)-1:
            batchData = {
                "w2v": numpy.array(w2vFeatures),
                "hc": numpy.array(hcFeatures),
                "labels": labels
            }
            with open(os.path.join(outputFolder, str(batch))+BATCH_FORMAT, "wb") as output:
                pickle.dump(batchData, output, pickle.HIGHEST_PROTOCOL)
            print("\rSaved batch "+str(batch), end="")
            w2vFeatures = []
            hcFeatures = []
            labels = []
            batch += 1

        if lines[i] in OMITED_DATA:
            continue
        wordData = lines[i][:-1].split(" ") # cut out '\n'
        word = wordData[DATA_FORMAT["word"]]
        w2vFeatures.append(w2vEx.extract(word))

        preWordData = lines[i-1].split(" ")
        posWordData = lines[i+1].split(" ")
        hcFeatures.append(hcEx.extract(word, posWordData[0]=="", preWordData[0]==""))

        labels.append(NE_LABEL[re.sub(r".*-", "", wordData[DATA_FORMAT["net"]])])
        
    print("\n")

def load_batch(dataFile):
    inFile = open(dataFile, "rb")
    batchData = pickle.load(inFile)
    inFile.close()
    return [batchData["w2v"], batchData["hc"]], batchData["labels"]

def train(model, folderTrain, modelName, weightName):
    batch_data_files = glob.glob(folderTrain+"/*"+BATCH_FORMAT)
    for batch_file in batch_data_files:
        X, Y = load_batch(batch_file)
        Y = CNNModel.convert_labels(Y, numClass=5)
        model.train(X, Y)
    model.save(modelName, weightName)

def test(model, folderTest, modelName, weightName):
    model.load(modelName, weightName)
    batch_data_files = glob.glob(folderTest+"/*"+BATCH_FORMAT)
    for batch_file in batch_data_files:
        X, Y = load_batch(batch_file)
        statistic = collections.Counter(Y)
        result = model.test(X, Y, labels=NE_LABEL.values())

        print("Dataset statistic:\n", 
            json.dumps({k:statistic[NE_LABEL[k]] for k in NE_LABEL.keys()}, indent=4))
        print("Confusion matrix:\n", 
            result["confusion_matrix"])
        print("F1-score for each class:\n", 
            json.dumps({k:result["f1_score"][NE_LABEL[k]] for k in NE_LABEL.keys()}, indent=4))
        print("F1-score micro average: ", 
            result["f1_score_micro"])
        print("F1-score macro average: ", 
            result["f1_score_macro"])
        # Y = CNNModel.convert_labels(Y, numClass=5)
        # print(model.test_auto(X, Y))

if __name__ == '__main__':
    # python main.py -m train -e 1 -d dat/eng.train -o dat/ -w dat/vectors.bin 
    # python main.py -m test -d dat/eng.testa -o dat/ -w dat/vectors.bin -c dat/weight_9_a1.h5 -a dat/model.json 
    # python main.py -m type -d dat/input.txt -o dat/ -w dat/vectors.bin -c dat/weight_9_a1.h5 -a dat/model.json 

    parser = argparse.ArgumentParser(description="main program to train or test cnn model")
    parser.add_argument("-m", "--mode", 
        help="[train, test, type] train data file/test data file", required=True)
    parser.add_argument("-d", "--data", help="data file", required=True)
    parser.add_argument("-o", "--output", help="output prefix", required=False, default="")
    parser.add_argument("-s", "--batch_size", 
        help="use for mode 'train', size of each batch file (default: 100)", 
        required=False)
    parser.add_argument("-e", "--epoch", 
        help="use for mode 'train', number of epoch (default: 5)", 
        required=False)
    parser.add_argument("-w", "--w2v_file", 
        help="word2vec pretrained model file", 
        required=True)
    parser.add_argument("-c", "--weight_file", 
        help="use for mode 'test' or 'type', cnn pretrained weight file", 
        required=False)
    parser.add_argument("-a", "--architech_file", 
        help="use for mode 'test' or 'type', cnn architech file", 
        required=False)


    args = parser.parse_args()

    option = args.mode.lower()
    
    batchSize = 100
    if args.batch_size:
        batchSize = int(args.batch_size)

    epoch = 5
    if args.epoch:
        epoch = int(args.epoch)

    rawDataFile = args.data
    outputFolder = args.output

    w2vEx = W2VExtractor()
    w2vEx.load(args.w2v_file)
    hcEx = HandcraftExtractor()

    model = CNNModel()

    if os.path.isdir(os.path.join(outputFolder, "batches")):
        shutil.rmtree(os.path.join(outputFolder, "batches"))

    os.mkdir(os.path.join(outputFolder, "batches"))

    if option == "train":
        convert_data(
            rawDataFile, os.path.join(outputFolder, "batches"), 
            w2vEx, hcEx, batchSize)
        for e in range(epoch):
            train(
                model, os.path.join(outputFolder, "batches"), 
                os.path.join(outputFolder, "model"+MODEL_FORMAT), 
                os.path.join(outputFolder, "w_"+str(e)+WEIGHT_FORMAT))
    elif option == "test":
        if not args.weight_file or not args.architech_file:
            print("Error: Missing pretrained model")
            sys.exit()
        convert_data(
            rawDataFile, os.path.join(outputFolder, "batches"), 
            w2vEx, hcEx)
        test(
            model, os.path.join(outputFolder, "batches"),
            args.architech_file,
            args.weight_file)
    elif option == "type":
        if not args.weight_file or not args.architech_file:
            print("Error: Missing pretrained model")
            sys.exit()
        model.load(args.architech_file, args.weight_file)

        inFile = open(rawDataFile, "r")
        outFile = open(rawDataFile+".res", "w")
        sentences = Tokenizer.tokenizeSentence(inFile.read())
        for sentence in sentences:
            words = Tokenizer.tokenizeWord(sentence)
            
            w2vFeatures = []
            hcFeatures = []
            for i in range(len(words)):
                w2vFeatures.append(w2vEx.extract(words[i]))
                hcFeatures.append(hcEx.extract(words[i], i==len(words)-1, i==0))

            X = [numpy.array(w2vFeatures), numpy.array(hcFeatures)]
            y = [numpy.argmax(v) for v in model.predict(X)]

            result = {}
            for i in range(len(words)):
                if y[i] != NE_LABEL["O"]:
                    result[words[i]] = NE_LOOKUP[y[i]]
            print(result)
            outFile.write(str(result)+"\n")
        inFile.close()
        outFile.close()


