import random
import json
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def cleanText(rawstr):
    lmtzr = WordNetLemmatizer()
    stmmr = PorterStemmer()
    stopWords = set(stopwords.words('english'))

    rawstr = rawstr.lower()
    newstr = ""
    for char in rawstr:
        if char.isalpha() or char == " ":
            newstr = newstr + char
        elif char == "'":
            newstr = newstr + " "
    newstr = newstr.split()
    result = [ ]
    for word in newstr:
        if len(word) < 2:
            continue
        if word in stopWords:
            continue
        result.append(stmmr.stem(lmtzr.lemmatize(word)))
    return result


def getItemData(filename = "citeulike-a/raw-data.csv"):
    f = open(filename)
    lines = f.readlines()[1:]
    f.close()
    cleaned = [ ]
    for line in lines:
        cleaned.append(cleanText(line))
    #
    f = open("word2int.json")
    word2int = json.loads(f.read())
    f.close()
    #
    result = np.zeros((len(cleaned),len(word2int)))
    print(result.shape)
    for i in range(len(cleaned)):
        for j in cleaned[i]:
            if j in word2int:
                result[i][word2int[j]] = 1.0
    return result


def getTrainTestData(filename = "citeulike-a/users.dat"):
    items = getItemData()
    f = open(filename)
    lines = f.readlines()
    f.close()
    clnd = [ ]
    ngtv = [ ]
    for line in lines:
        nums = line.split()[1:]
        nums = [ int(n) for n in nums ]
        negative = set()
        while len(negative) < len(nums):
            rnd = random.randint(0, 16979)
            if rnd in nums:
                continue
            negative.add(rnd)
        random.shuffle(nums)
        clnd.append(nums)
        ngtv.append(list(negative))
    trainSet = [ ]
    devSet = [ ]
    testSet = [ ]
    for i in range(len(clnd)):
        devSet.append((i, items[clnd[i][0]], 1.0))
        devSet.append((i, items[ngtv[i][0]], 0.0))
        testSet.append((i, items[clnd[i][1]], 1.0))
        testSet.append((i, items[ngtv[i][1]], 0.0))
        for j in range(2, len(clnd[i])):
            trainSet.append((i, items[clnd[i][j]], 1.0))
            trainSet.append((i, items[ngtv[i][j]], 0.0))
    random.shuffle(trainSet)
    random.shuffle(devSet)
    random.shuffle(testSet)
    return (trainSet, devSet, testSet)


