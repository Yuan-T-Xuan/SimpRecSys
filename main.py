import numpy as np
from citeulikeDataPrep import getTrainTestData
from NNTrain import trainModel

trainSet, devSet, testSet = getTrainTestData()
input0_train = np.array([ d[0] for d in trainSet ], dtype='int32')
input1_train = np.array([ d[1] for d in trainSet ])
output_train = np.array([ d[2] for d in trainSet ])
input0_train = np.reshape(input0_train, (-1,1))
output_train = np.reshape(output_train, (-1,1))

input0_dev = np.array([ d[0] for d in devSet ], dtype='int32')
input1_dev = np.array([ d[1] for d in devSet ])
output_dev = np.array([ d[2] for d in devSet ])
input0_dev = np.reshape(input0_dev, (-1,1))
output_dev = np.reshape(output_dev, (-1,1))

input0_test = np.array([ d[0] for d in testSet ], dtype='int32')
input1_test = np.array([ d[1] for d in testSet ])
output_test = np.array([ d[2] for d in testSet ])
input0_test = np.reshape(input0_test, (-1,1))
output_test = np.reshape(output_test, (-1,1))

input0_devf = open("input0_devf.npy", 'wb')
np.save(input0_devf, input0_dev)
input0_devf.close()
input1_devf = open("input1_devf.npy", 'wb')
np.save(input1_devf, input1_dev)
input1_devf.close()
output_devf = open("output_devf.npy", 'wb')
np.save(output_devf, output_dev)
output_devf.close()

input0_testf = open("input0_testf.npy", 'wb')
np.save(input0_testf, input0_test)
input0_testf.close()
input1_testf = open("input1_testf.npy", 'wb')
np.save(input1_testf, input1_test)
input1_testf.close()
output_testf = open("output_testf.npy", 'wb')
np.save(output_testf, output_test)
output_testf.close()

# for debug
print(input0_train.shape)
print(input1_train.shape)
print(output_train.shape)

trainModel(BoWSize=5000, EncodedSize=200, input_dim=5551, output_dim=200, inputs=(input0_train,input1_train), output=output_train, epochs=20, batch_size=256, outfile_name="tmp-out.h5")

