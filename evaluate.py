from keras.models import load_model
import numpy as np
from citeulikeDataPrep import getTrainTestData

trainSet, devSet, testSet = getTrainTestData()
del trainSet
input0_dev = np.array([ d[0] for d in devSet ], dtype='int32')
input1_dev = np.array([ d[1] for d in devSet ])
output_dev = np.array([ d[2] for d in devSet ])
input0_dev = np.reshape(input0_dev, (-1,1))
output_dev = np.reshape(output_dev, (-1,1))

model = load_model('tmp-out.h5')
pred = model.predict([input0_dev, input1_dev])

"""
trainSet, devSet, testSet = getTrainTestData()
input0_train = np.array([ d[0] for d in trainSet[:10000] ], dtype='int32')
input1_train = np.array([ d[1] for d in trainSet[:10000] ])
output_train = np.array([ d[2] for d in trainSet[:10000] ])
input0_train = np.reshape(input0_train, (-1,1))
output_train = np.reshape(output_train, (-1,1))

input0_dev = input0_train
input1_dev = input1_train
output_dev = output_train

model = load_model('tmp-out.h5')
pred = model.predict([input0_dev, input1_dev])
"""

p0r0 = 0
p1r0 = 0
p0r1 = 0
p1r1 = 0
for i in range(10000):
    if output_dev[i][0] > 0.5:
        if pred[0][i][0] > 0.1:
            p1r1 += 1
        else:
            p0r1 += 1
    else:
        if pred[0][i][0] > 0.1:
            p1r0 += 1
        else:
            p0r0 += 1

