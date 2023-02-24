
# Introduction

This script is designed to train an object location model that can predict the location of an object in an image. The model uses a dataset of images with labeled object coordinates, and it performs data augmentation to increase the size of the dataset. The script uses a convolutional neural network with the Inception V3 architecture, and it trains the model using the mean squared error (MSE) loss function.

# Requirements

The following libraries are required to run this script:

- os
- csv
- glob
- numpy
- pandas
- opencv-python
- torch
- torch_lr_finder
- scikit-image
- tqdm
- Pillow

If the mentioned packages are not installed on your system, You can insatll them through the following command:

'''

python my_script.py --input_file input.txt --output_file output.txt

'''


# Data Augmentation

The data augmentation is performed on the input images to increase the size of the dataset. The following augmentations are applied to each image:

- Raw image
- Speckle noise
- Gaussian noise
- Salt and pepper noise
- Flip, rotation, and translation

The augmented images are saved in a destination folder, and the corresponding object coordinates are saved in a CSV file.

# Dataset

The script reads the object coordinates from a CSV file that contains the filename of each image and the corresponding X and Y coordinates of the object in the image. The script splits the dataset into training, validation, and testing sets, and it creates a PyTorch Dataset object for each set.

# Model Architecture

The model uses the Inception V3 architecture as the backbone, and it outputs the predicted X and Y coordinates of the object in the image. The model is trained using the MSE loss function.

# Learning Rate Finder

The learning rate finder is used to find the optimal learning rate for the model. The LR_Finder class takes the model and the training dataset as input, and it returns the suggested learning rate.

# Model Training

The model is trained using the Adam optimizer with the suggested learning rate. The training loop runs for the specified number of epochs, and the model with the best validation loss is saved. The script outputs the training and validation loss for each epoch.

# Usage

To use this script, you need to set the following parameters:

'**source_folder**': the folder that contains the original images and the CSV file with the object coordinates

'**destination_folder**': the folder where the augmented images and the new CSV file will be saved

'**model_name**': the name of the pre-trained model to use as the backbone (e.g., "inception_v3")

'**batch_size**': the batch size for training the model

'**epochs**': the number of epochs to train the model

Once the parameters are set, you can run the script. The script will perform the data augmentation, split the dataset into training, validation, and testing sets, train the model, and save the best model.



