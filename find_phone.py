import cv2
import argparse
import torch
import timm
import os


class ObjLocModel(torch.nn.Module):
    def __init__(self, model_name, num_classes=2):
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


def preprocess_image(image_path):
    # Load the test image and preprocess it
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
    img = img.unsqueeze(0)
    return img

def detect_phone(image_path, model, device):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Use the pre-trained model to predict the location of the phone
    with torch.no_grad():
        out = model(img)
    x, y = out[0][0], out[0][1]

    # Print the normalized coordinates of the phone detected in the test image, rounded to four decimal places
    print(f"{x:.4f} {y:.4f}")

def main(args):
    # Set up the object detection model
    model_checkpoint_path = "trained_model.pt"
    model_name = "densenet121"
    device = "cpu"
    image_path = os.path.expanduser(args.image_path)

    # Load the trained model
    model = ObjLocModel(model_name)
    model.to(device)
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Detect the phone in the image
    detect_phone(image_path, model, device)


if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description='Find phone in an image')
    parser.add_argument('image_path', type=str, help='Path to the test image')
    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args)
