from __future__ import print_function

import os
import word2vec
import numpy
import string

class W2VExtractor:
    def __init__(self, originData=None, w2vModelPath="vectors.w2v", vectorSize=100):
        self.__model = None
        self.__vectorSize = vectorSize
        if type(originData) is str:
            word2vec.word2vec(
                originData, 
                w2vModelPath, 
                size=vectorSize, 
                verbose=True)
            self.__model = word2vec.load(w2vModelPath)

    def load(self, wordVectorFile):
        self.__model = word2vec.load(wordVectorFile)
        self.__vectorSize = self.__model.vectors.shape[1]

    def getVectorSize(self):
        return self.__vectorSize

    def extract(self, word):
        if self.__model is None:
            print("Error: The model is None")
            return None

        if not word:
            print("Error: The word is None")
            return None
        wordVector = None
        try:
            wordVector = self.__model[word.lower()]
        except:
            wordVector = numpy.zeros((self.__vectorSize,))
        return wordVector.reshape((1, self.__vectorSize))

class HandcraftExtractor:
    PUNCT = set(string.punctuation)

    def extract(self, word, isFirst=False, isLast=False):
        return numpy.array([
            len(word), 
            len(set(word)&HandcraftExtractor.PUNCT),
            int(word[0].islower()),
            int(word.islower()),
            int(unicode(word, 'utf-8').isnumeric()),
            int(isFirst),
            int(isLast)
        ]).reshape((1, 7))





