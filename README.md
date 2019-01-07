# AutomatedLipReader

#This project hoped to utilize a Convolutional LSTM model to analyze phenomes through visual analysis of lip movement. Overall the dataset was too small and caused severe overfitting on the training data which led to large discrepancies on the testing data. A much larger and more standardized dataset would be needed to make progress on the issue

#We wanted convolutional neural networks to have accurate image recognition but primarily felt the LSTM was key to use previous frame memory and recognize overall movement of the lip to determine phoneme movement

#VisualizeLip is a combination of VideoToImageConverter and mouthV3 in order to easily convert videos in the dataset to input data consisting of the pictures of lip segments

#CNN_LSTM codes our training model

#The model available can still be easily used to analyze videos for other purposes by removing facial or lip segmenting software.
