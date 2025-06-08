import torch

from number_img_detection.train_model import Digit_Classifier
from PIL import Image
from torchvision import transforms

# Load Model
model = Digit_Classifier()
model.load_state_dict(torch.load("/Users/mahmoudhachem/Personal_Projects-Learning/number_img_detection/digit_classifier.pth"))

# Transform + Load Image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # turn to grayscale img
    transforms.Resize((28, 28)), # Resize to 28x28 like training data
    transforms.ToTensor(), # convert to Tensor
    transforms.Normalize((0.1307, ), (0.3081, )) # normalize like MNIST
])
img_path = '/Users/mahmoudhachem/Downloads/printable-number-0-232x300.jpg'
img = Image.open(img_path)
image = transform(img)
image = image.unsqueeze(0)
# pytorch expects models in 4D shape. .unsqueeze() adds a batch dimension.

with torch.no_grad():
    output = model(image) # shape: (1, 10)
    predicted = torch.argmax(output, dim=1)
    # argsMax:
        # finds the number in the tensor with the highest value
        # values represent model's confidence (logits) for each digit class
    print(f"Predicted digit: {predicted.item()}")
