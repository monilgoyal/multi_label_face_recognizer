import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
model_name = input("Give a unique model name: ")
# Get the training data we previously made
data_path = './faces/'
labels = [f for f in listdir(data_path) if f[0] != '.']
dirfiles = [[f for f in listdir(
    data_path+l) if isfile(join(data_path+l, f))] for l in labels]


# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for folder, folder_name in enumerate(labels):
    for i, files_name in enumerate(dirfiles[folder]):
        image_path = data_path+folder_name+'/' + files_name
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i+folder*100)

model = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model
model.train(np.asarray(Training_Data), np.asarray(Labels))

for folder, folder_name in enumerate(labels):
    for i, files_name in enumerate(dirfiles[folder]):
        model.setLabelInfo(i+folder*100, folder_name)

model.save(f'./model/{model_name}.yml')
print("Model trained and saved sucessefully")
