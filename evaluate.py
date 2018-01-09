from keras.models import load_model
import numpy as np
from citeulikeDataPrep import getTrainTestData

input0_devf = open("input0_devf.npy", 'rb')
input1_devf = open("input1_devf.npy", 'rb')
output_devf = open("output_devf.npy", 'rb')
input0_testf = open("input0_testf.npy", 'rb')
input1_testf = open("input1_testf.npy", 'rb')
output_testf = open("output_testf.npy", 'rb')

input0_dev = np.load(input0_devf)
input1_dev = np.load(input1_devf)
output_dev = np.load(output_devf)

model = load_model('tmp-out.h5')
pred = model.predict([input0_dev, input1_dev])

threshold = 0.00
X = [0.0] # false positive rate
Y = [0.0] # true positive rate
while threshold < 1.00:
    threshold += 0.01
    p0r0 = 0
    p1r0 = 0
    p0r1 = 0
    p1r1 = 0
    for i in range(11102):
        if output_dev[i][0] > 0.5:
            if pred[0][i][0] > 1.00 - threshold:
                p1r1 += 1
            else:
                p0r1 += 1
        else:
            if pred[0][i][0] > 1.00 - threshold:
                p1r0 += 1
            else:
                p0r0 += 1
    X.append( p1r0 / (p1r0 + p0r0) )
    Y.append( p1r1 / (p1r1 + p0r1) )

AUC = 0.0
for i in range(1, len(X)):
    AUC += (Y[i-1] + Y[i]) * (X[i] - X[i-1])
print("AUC: " + str(AUC/2.0))

