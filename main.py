import numpy as np
from citeulikeDataPrep import getTrainTestData
from NNTrain import trainModel

trainSet, devSet, testSet = getTrainTestData()
input0_train = np.array([ d[0] for d in trainSet ], dtype='int32')
input1_train = np.array([ d[1] for d in trainSet ])
output_train = np.array([ d[2] for d in trainSet ])
input0_train = np.reshape(input0_train, (-1,1))
output_train = np.reshape(output_train, (-1,1))

# for debug
print(input0_train.shape)
print(input1_train.shape)
print(output_train.shape)

trainModel(BoWSize=5000, EncodedSize=200, input_dim=5551, output_dim=200, inputs=(input0_train,input1_train), output=output_train, epochs=20, batch_size=128, outfile_name="tmp-out.h5")

