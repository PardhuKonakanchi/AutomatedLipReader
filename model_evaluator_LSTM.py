from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from resizeList import addFrames
import cv2
import glob
import pickle
import os
import letter_label_pairs
import histConverter
import numpy as np


time_step = 10
unscaled_frame_size = 128
frame_size = 64
numlabels = 30
unscaled_frame_size_tuple = (unscaled_frame_size, unscaled_frame_size)
frame_size_tuple = (frame_size, frame_size)

face_cascade = cv2.CascadeClassifier('C:\\Users\\anish\\Desktop\\SiemensStuff\\haarcascade_frontalface_default.xml')

path_to_sample = 'C:\\Users\\single_evaluation_data_64\\bound_lips_64\\TestSetStanford_64_fixed\\I am sorry'
snapshot_model = 'model_09152017_trained_snapshot_v38_bound_lips_low_epocs.h5'
path_to_snapshot_folder = 'C:\\Users\\snapshots\\CNN_LSTM\\'
img = glob.glob(path_to_sample + '\\*.jpg')
img = addFrames(img = img, timesteps = time_step)
img.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
numFrames = len(img)
tmp_data = np.zeros((int(numFrames/time_step), time_step, frame_size, frame_size, 3))

d5_count = 0
count = 0
for image in img:
    frame = cv2.imread(image)
    tmp_data[d5_count, int(count%(time_step))] = frame
    if (count+1)%(time_step) == 0:
        d5_count+=1
    count += 1

evaluation_sample = tmp_data
evaluation_sample = evaluation_sample.astype('uint8')
evaluation_sample = evaluation_sample/np.max(evaluation_sample)

batch_size = len(img)
model = load_model(path_to_snapshot_folder + snapshot_model)
evaluation_output = model.predict(x=evaluation_sample, batch_size = batch_size, verbose=0)

# np.savetxt('C:\\Users\\anish\\Desktop\\SiemensStuff\\raw_prediction.txt', evaluation_output, fmt='%10.3f')
print(evaluation_output.shape)
evaluation_output = evaluation_output.reshape((evaluation_output.shape[0]*evaluation_output.shape[1], numlabels))
print(evaluation_output.shape)
max_indices = np.argmax(evaluation_output, axis = 1)
evaluation_output_conv = np.zeros_like(evaluation_output)
evaluation_output_conv[np.arange(len(evaluation_output)), max_indices] = 1

# max_indices = np.argmax(evaluation_output, axis = 1)
# for i in evaluation_output:
# 	evaluation_output[i]
# 	evaluation_output[i, maxindeces = ()]
output_conv_array = np.arange(len(letter_label_pairs.Label))+1 #this array is used to convert the model output predictions into the enum values that are stored in leter_label_pairs
number_output = np.dot(evaluation_output_conv, output_conv_array)

letter_output = []

iterator = 0
for i in number_output:
	for l in letter_label_pairs.Label:
		if l.value == number_output[iterator]:
			letter_output.append(l.name)
	iterator = iterator + 1

count = 0
for i in letter_output:
	if i == 'space':
		letter_output[count] = ' '
	count+=1


print(letter_output)
final_result = histConverter.convCharArray(letter_output)
output_file_temp = open('C:\\Users\\output_test.txt', 'w')
for line in final_result:
	output_file_temp.write(str(line))
output_file_temp.close()
print(final_result)




