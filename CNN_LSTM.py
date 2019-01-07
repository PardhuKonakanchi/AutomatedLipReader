from keras.models import Model, load_model
from keras.layers import LSTM, Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
import numpy as np
import preprocessData
import math
from keras.models import Sequential

######## NOTE #############
# When training, remember to change the snapshot file name each time to match the date
snapshot_prefix = 'C:\\Users\\snapshots\\CNN_LSTM\\'
snapshot_filename = 'model_09072017_trained_snapshot_v13_no_drop.h5'

# set model parameters
batchSize = 50
epochSize = 100
kernel = 3
pool = 2
conv1 = 32
drop1 = .25
drop2 = .5
hidden1 = 256

# image dimensions
imageHeight = 64
imageLength = 64
imageDepth = 3
timesteps = 10

inputShape = (timesteps, imageHeight, imageLength, imageDepth)

# number of character labels
numlabels = 30

# preprocess and format Data
TotalData,TotalLabels = preprocessData.formatData(net_type="cnn_lstm")
print(TotalData.shape[0])
print(TotalLabels.shape[0])
splitpoint_a = math.floor(TotalData.shape[0]*0.7)
splitpoint_b = math.floor(TotalData.shape[0]*0.9)
data = TotalData[0:splitpoint_a]
labels = TotalLabels[0:splitpoint_a]
ValidationData = TotalData[splitpoint_a:splitpoint_b]
ValidationLabels = TotalLabels[splitpoint_a:splitpoint_b]
TestData = TotalData[splitpoint_b:]
TestLabels = TotalLabels[splitpoint_b:]

print(data.shape)
data.astype('uint8')
data = data.astype('uint8')
ValidationData = ValidationData.astype('uint8')
data = data / np.max(data)
ValidationData = ValidationData / np.max(ValidationData)




# setup convolutionalLSTM architecture
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), padding='valid'),input_shape=inputShape))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
print (model.output_shape)
model.add(TimeDistributed(Conv2D(64, (3, 3), padding='valid')))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
# model.add(Dropout(0.25))
# model.add(TimeDistributed(Conv2D(128, (3, 3), padding='valid')))
# model.add(Activation('relu'))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
print(model.output_shape)

model.add(TimeDistributed(Flatten()))
print (model.output_shape)
model.add(LSTM(units=timesteps, return_sequences=True))
print (model.output_shape)
# model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(numlabels)))
print (model.output_shape)
model.add(Activation('sigmoid'))
print (model.output_shape)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(data, labels, batch_size=batchSize, epochs=epochSize, verbose=1, validation_data=(ValidationData, ValidationLabels))
score = model.evaluate(TestData, TestLabels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# saves keras model, with architecture, weights, model fitting configuration, etc, as a .h5 file
model.save(snapshot_prefix + snapshot_filename)
