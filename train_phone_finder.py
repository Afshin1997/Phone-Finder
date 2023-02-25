import argparse
import os
import csv
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch_lr_finder import LRFinder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import timm
from skimage.util import random_noise
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F


class DataAugment:
    def __init__(self, source_path, destination_folder, image_file, csv_file, image_number, row,
                 p_flip=0.5, max_rotation=15, max_translation=0.1):
        """
        Initializes the data augmentation class with the specified source and destination folders,
        image and CSV files, and other parameters such as the probability of flipping an image,
        the maximum rotation angle, and the maximum translation value.
        """

        self.source_path = source_path
        self.destination_folder = destination_folder
        self.image_number = image_number
        self.image_file = image_file
        self.row = row
        self.row[1] = float(self.row[1])
        self.row[2] = float(self.row[2])
        self.csv_file = csv_file

        # Open the CSV file for writing
        self.csv_handle = open(self.csv_file, "a", newline="")
        self.writer = csv.writer(self.csv_handle, delimiter=",")
        self.p_flip = p_flip
        self.max_rotation = max_rotation
        self.max_translation = max_translation
        self.image = cv2.imread(os.path.join(self.source_path, self.image_file))

    def raw_img(self):
        """
        Saves the original image to the destination folder and writes the image numbers and CSV rows accordingly.
        """
        cv2.imwrite(os.path.join(self.destination_folder, f"{self.image_number}.jpg"), self.image)
        self.row[0] = f"{self.image_number}.jpg"
        self.writer.writerow(self.row)
        self.image_number += 1

    def speckle(self):
        """
        Applies speckle noise to the image, saves the results to the destination folder, and writes the image numbers
        and CSV rows accordingly.
        """
        image = random_noise(self.image, mode="speckle", mean=0, var=0.3, clip=True)
        image = image * 255
        cv2.imwrite(os.path.join(self.destination_folder, f"{self.image_number}.jpg"), image)
        self.row[0] = f"{self.image_number}.jpg"
        self.writer.writerow(self.row)
        self.image_number += 1

    def gaussian(self):
        """
        Applies Gaussian noise to the image, saves the results to the destination folder, and writes the image numbers
        and CSV rows accordingly.
        """
        image = random_noise(self.image, mode="gaussian", mean=0, var=0.03)
        image = image * 255
        cv2.imwrite(os.path.join(self.destination_folder, f"{self.image_number}.jpg"), image)
        self.row[0] = f"{self.image_number}.jpg"
        self.writer.writerow(self.row)
        self.image_number += 1

    def sandp(self):
        """
        Applies salt and pepper noise to the image, saves the result to the destination folder, and writes the image
        number and CSV row accordingly.
        """
        image = random_noise(self.image, mode="s&p", amount=0.05)
        image = image * 255
        cv2.imwrite(os.path.join(self.destination_folder, f"{self.image_number}.jpg"), image)
        self.row[0] = f"{self.image_number}.jpg"
        self.writer.writerow(self.row)
        self.image_number += 1

    def flip_rot_trans(self):
        """
        Applies a combination of horizontal flipping, randomly rotation, and randomly translate to the images, saves the
        results to the destination folder, and writes the images numbers  and CSV rows accordingly.
        """
        image = F.hflip(Image.fromarray(self.image))
        self.row[1] = 1 - self.row[1]

        angle = torch.randint(-self.max_rotation, self.max_rotation + 1, (1,)).item()
        image = F.rotate(image, angle)

        dx = torch.randint(-int(self.max_translation * image.size[0]), int(self.max_translation * image.size[0]) + 1,
                           (1,)).item()
        dy = torch.randint(-int(self.max_translation * image.size[1]), int(self.max_translation * image.size[1]) + 1,
                           (1,)).item()
        image = F.affine(image, angle=0, translate=(dx, dy), scale=1, shear=0)
        self.row[1] += dx / image.size[0]  # adjust the x-coordinate accordingly
        self.row[2] += dy / image.size[1]

        self.row[1] = round(self.row[1], 4)
        self.row[2] = round(self.row[2], 4)

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.destination_folder, f"{self.image_number}.jpg"), image)
        self.row[0] = f"{self.image_number}.jpg"
        self.writer.writerow(self.row)
        self.image_number += 1

    def get_image_number(self):
        # Return the number of images created so far
        return self.image_number

    def __del__(self):
        # Close the CSV file handle
        self.csv_handle.close()


class DatasetAugmentation:
    def __init__(self, source_path, destination_path):
        """
        Initializes the DatasetAugmentation object with the source and destination path, CSV file path, image number,
        and header for the CSV file.
        """
        self.source_path = source_path
        self.destination_path = destination_path
        self.csv_file = os.path.join(destination_folder, "labels.csv")
        self.labels_file = os.path.join(source_path, "labels.txt")
        self.image_number = 0
        self.header = ["File_name", "X_coordinate", "Y_coordinate"]
        self.jpg_files = glob.glob(os.path.join(source_path, '*.jpg'))

    def run(self):
        """
        It creates the destination folder if it doesn't exist and writes the header to the CSV file. Also, this function
        only processes the JPG files in the source folder, and finally, it augments the images and writes the results to
        the destination folder and CSV file.
        """
        if not os.path.exists(self.destination_path):
            os.mkdir(self.destination_path)
            print("The destination path is created!!!!")

        with open(self.csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.header)

        for filename in os.listdir(self.source_path):
            if not filename.endswith('.jpg'):
                continue

            with open(self.labels_file, 'r') as f:
                reader = csv.reader(f, delimiter=" ")

                for row in reader:
                    if row[0] == filename:
                        augmentor = DataAugment(self.source_path, self.destination_path, filename, self.csv_file,
                                                self.image_number, row)
                        augmentor.raw_img()
                        augmentor.speckle()
                        augmentor.gaussian()
                        augmentor.sandp()
                        augmentor.flip_rot_trans()

                        self.image_number = augmentor.get_image_number()


class ObjLocDataset(torch.utils.data.Dataset):
    """
    This is a PyTorch dataset class that loads images and corresponding object center positions from a CSV file.
    The class takes a DataFrame and a destination folder as inputs.
    """
    def __init__(self, df, destination_folder):
        self.df = df
        self.destination_folder = destination_folder

    def __len__(self):
        # The length of the dataset is the length of the input DataFrame.
        return len(self.df)

    def __getitem__(self, idx):
        """
        loads the image at the specified row, converts it to a PyTorch tensor, and normalizes it. It also extracts
        the object center position from the row and returns both the image tensor and the center position as a tuple.
        """
        row = self.df.iloc[idx]

        X_pos = row.X_coordinate
        Y_pos = row.Y_coordinate

        center_pos = [X_pos, Y_pos]

        img_path = os.path.join(self.destination_folder, row.File_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0  # hwc ==> chw
        center_pos = torch.Tensor(center_pos)

        return img, center_pos


class ObjLocModel(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        """
        It initializes the object with the given model_name and number of num_classes, creating the backbone of the model.
        """
        super(ObjLocModel, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, images, gt_center_poses=None):
        """
        Takes in an array of images and returns an array of center positions as predicted by the model. If ground truth
        center positions are given, calculates the mean squared error loss between the predicted and true center
        positions and returns both the predicted center positions and the loss.
        """
        center_poses = self.backbone(images)

        if gt_center_poses != None:
            loss = torch.nn.MSELoss()(center_poses, gt_center_poses)
            return center_poses, loss

        return center_poses

class ModelTraining:
    def __init__(self, destination_folder, model_name, batch_size, epochs):
        """
        It initializes the model training with the source folder, destination folder, model name, batch size, and number
        of epochs for training.
        """
        self.destination_folder = destination_folder
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs

    def train_model(self, model, train_loader, valid_loader, optimizer, device, epochs):
        """
        It takes in the model, data loader for training, data loader for validation, optimizer, device, and number of
        epochs. The method loops through the epochs, calling the _train_step and _evaluate_step methods to train and
        validate the model for each epoch, respectively. It saves the weights if the current validation loss is lower
        than the best validation loss.
        """
        best_valid_loss = np.Inf

        for n_epochs in range(epochs):
            train_loss = self._train_step(model, train_loader, optimizer, device)
            valid_loss = self._evaluate_step(model, valid_loader, device)

            checkpoint = {
                'epoch': n_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': valid_loss
            }

            if valid_loss < best_valid_loss:
                model_checkpoint_path = "trained_model.pt"
                torch.save(checkpoint, model_checkpoint_path)

                print("WEIGHTS ARE SAVED")
                best_valid_loss = valid_loss

            print(f"Epoch: {n_epochs + 1} train loss: {train_loss} valid loss: {valid_loss}")

    def _train_step(self, model, train_loader, optimizer, device):
        """
        It takes in the model, data loader for training, optimizer, and device. The method loops through the training data,
         performs forward and backward propagation, and returns the average training loss for the epoch.
        """
        total_loss = 0.0
        model.train()

        for data in tqdm(train_loader):
            images, gt_center_poses = data
            images, gt_center_poses = images.to(device), gt_center_poses.to(device)

            center_poses, loss = model(images, gt_center_poses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _evaluate_step(self, model, valid_loader, device):
        """
        It takes in the model, data loader for validation, and device. The method loops through the validation data and
        returns the average validation loss for the epoch.
        """
        total_loss = 0.0
        model.eval()

        with torch.no_grad():
            for data in tqdm(valid_loader):
                images, gt_center_poses = data
                images, gt_center_poses = images.to(device), gt_center_poses.to(device)

                center_poses, loss = model(images, gt_center_poses)
                total_loss += loss.item()

            return total_loss / len(valid_loader)


if __name__ == "__main__":

    # Define command line arguments for the data folder path using argparse.
    parser = argparse.ArgumentParser(description='Train phone finder')
    parser.add_argument('data_folder', type=str, help='Path to the folder with labeled images and labels.txt')
    args = parser.parse_args()

    # Check for the availability of GPU and select the device (GPU or CPU).
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    lr = 0.00007
    batch_size = 4
    epochs = 200
    model_name = "densenet121"

    source_path = os.path.expanduser(args.data_folder)
    destination_folder = "new_dataset"

    augment = DatasetAugmentation(source_path, destination_folder)
    augment.run()

    csv_file = os.path.join(destination_folder, "labels.csv")
    df = pd.read_csv(csv_file)

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(valid_df, test_size=0.5, random_state=42)

    train_set = ObjLocDataset(train_df, destination_folder)
    valid_set = ObjLocDataset(valid_df, destination_folder)
    test_set = ObjLocDataset(test_df, destination_folder)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    model = ObjLocModel(model_name, num_classes=len(train_df.axes[1]) - 1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Model_training = ModelTraining(destination_folder, model_name, batch_size, epochs)
    Model_training.train_model(model, train_loader, valid_loader, optimizer, device, epochs)