# AutomatedLipReader

#This project was completed last year by me and 2 peers and being uploaded now for viewing purposes. The project hoped to utilize a Convolutional LSTM model to analyze phenomes through visual analysis of lip movement. Overall the dataset was too small and caused severe overfitting on the training data which led to large discrepancies on the testing data. A much larger and more standardized dataset would be needed to make progress on the issue

# Data Collection

#The dataset consisted of video segments of me speaking all English phenomes which in total consisted of 30 classifications including a "space". I would usually speak for about 20 seconds into a camcorder with 30fps and these segments were split into each total phenome loop. The dataset is not uploaded here due to privacy issues.

# Data Preprocessing

#The preprocessing of the data went through several stages and was the primarly bulk of the project writing. Initially the videos were just converted to images and properly split to be sent to the model. After realizing this including unnecessary resources the model is training on, we decided to run the program through facial segmentation to exclude the background. Furthermore we decided to use lip segmentation which increased accuracy on average 5%. However, the preprocessing through lip segmentation was computationally expensive, taking hours to process through all the videos. For the goal of immediate lip reading in real life scenarios, facial recognition provides a better compromise without sacrificing too much accuracy. However, it was hard to test this differences once we realized the overall issue with our dataset as outlined earlier.

# Training Model

#We wanted convolutional neural networks to have accurate image recognition but due to low initial accuracy decided to implement primarily LSTMs. This was because they could use previous frame memory and recognize overall movement of the lip to determine phoneme classification much more accuractely than single frame image analysis.

#VisualizeLip is a combination of VideoToImageConverter and mouthV3 in order to easily convert videos in the dataset to input data consisting of the pictures of lip segments

#CNN_LSTM codes our training model

#The model available can still be easily used to analyze videos for other purposes by removing facial or lip segmenting processing.
