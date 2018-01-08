from citeulikeDataPrep import cleanText
import json
import random

filename = "citeulike-a/raw-data.csv"
f = open(filename)
lines = f.readlines()[1:]
f.close()
cleaned = [ ]
for line in lines:
    cleaned.append(cleanText(line))
frequency = dict()
for i in cleaned:
    for j in i:
        if not j in frequency:
            frequency[j] = 1
        else:
            frequency[j] += 1
l = [ ]
for i in frequency.items():
    l.append(i)
l.sort(key=(lambda x : x[1]), reverse=True)
l = [ w[0] for w in l[:5000]]
random.shuffle(l)
#
word2int = dict()
for i in l:
    word2int[i] = len(word2int)
outf = open("word2int.json", 'w')
outf.write(json.dumps(word2int))
outf.close()

