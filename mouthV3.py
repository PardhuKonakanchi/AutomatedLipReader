import cv2
import os
import glob
import pickle
import numpy as np

mouth_cascade = cv2.CascadeClassifier()# input path to downloaded xml file on haarcascade mouth detection
face_cascade = cv2.CascadeClassifier() #input path to downloaded xml file on haarcascade face deteciton

unscaled_frame_size = 128
frame_size = 64
unscaled_frame_size_tuple = (unscaled_frame_size, unscaled_frame_size)
frame_size_tuple = (frame_size, frame_size)
ds_factor = 1
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

pathTofolder = 'C:\\Users\\train_data\\raw_imgs_1440x1080\\train_data_raw_4\\tmp\\[space]' #newData is the file with the images in it
# onlyfiles = [f for f in listdir(pathTofolder) if isfile(join(pathTofolder,f))]

output_path = 'C:\\Users\\train_data\\bound_face_64x64\\train_data_bound_lips_v6\\'

def createData():
    count = 0
    for fn in os.listdir(pathTofolder):
        if fn == 'labels_combined.txt':
                continue
        new_img_folder = os.path.splitext(fn)[0]
        # if new_img_folder == "aa" or "ah":
        #     new_img_folder = "a"
        # elif new_img_folder == "eh":
        #     new_img_folder = "e"
        # elif new_img_folder == "ii":
        #     new_img_folder = "i"
        # elif new_img_folder == "oh":
        #     new_img_folder = "o"
        # elif new_img_folder == "uh":
        #     new_img_folder = "u"


        new_path = output_path + new_img_folder
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        tmpFileName = os.path.join(pathTofolder, fn)
        img = glob.glob(tmpFileName + '\\*.jpg')
        for image in img:
            if image is None:
                continue
            frame = cv2.imread(image)
            # frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face = face_cascade.detectMultiScale(gray, 1.3, 5)

            for(x,y,w,h) in face:
                # gray = gray[y:y+h, x:x+w]
                color = frame[y:y+h, x:x+w]
                color = cv2.resize(color, frame_size_tuple, 0, 0, interpolation=cv2.INTER_AREA)
                cv2.imwrite(new_path + '\\image%d.jpg' % count, color)
                break
                # mean_val = np.mean(color)
            # cv2.imshow('Bound lips', color)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # break
            count += 1
        print ("Phonetic sound " + fn + " created")

        # cv2.imshow("mouth detector", color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    # for (x,y,w,h) in mouth_rects:
    #     y = int(y - 0.15*h)
    #     # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    #     frame = frame[y:y+h, x:x+w]
    #     break

    # cv2.imshow('Mouth Detector', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
def checkTrainingData():
    for fn in os.listdir(pathTofolder):
        if fn == 'labels_combined.txt':
                continue
        tmpFileName = os.path.join(pathTofolder, fn)
        img = glob.glob(tmpFileName + '\\*.jpg')
        print(fn + ': ' + len(img))

createData()