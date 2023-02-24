import cv2
import numpy as np
import argparse

# Define the command line arguments
parser = argparse.ArgumentParser(description='Find phone in an image')
parser.add_argument('image_path', type=str, help='Path to the test image')
args = parser.parse_args()

# Load the pre-trained model
model_path = "internship4.pt"
model = torch.load(model_path)
model.eval()


# Load the test image and preprocess it
img = cv2.imread(args.image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
img = img.unsqueeze(0)

# Use the pre-trained model to predict the location of the phone
with torch.no_grad():
    out = model(image)
x, y = out[0], out[1]

# Print the normalized coordinates of the phone detected in the test image
print(f"{x.item()} {y.item()}")
