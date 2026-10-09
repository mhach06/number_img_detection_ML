import torch
from train_model import Digit_Classifier
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import argparse

if __name__ == '__main__':
    # Add arguments for user to input
    parser = argparse.ArgumentParser(description="Test the digit classifier on an image")
    parser.add_argument('--image', type=str, required=True, help='Path to the image you want to test')
    parser.add_argument('--model', type=str, default='digit_classifier.pth', help='Path to the saved model weights')
    args = parser.parse_args()

    # Load Model
    model = Digit_Classifier()
    model.load_state_dict(torch.load(args.model))
    model.eval()  # Set model to evaluation mode

    # Transform + Load Image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # turn to grayscale img
        transforms.Resize((28, 28)), # Resize to 28x28 like training data
        transforms.ToTensor(), # convert to Tensor
        transforms.Normalize((0.1307, ), (0.3081, )) # normalize like MNIST
    ])
    
    # Use the argument path instead of hardcoded path
    img = Image.open(args.image)
    image = transform(img)
    image = image.unsqueeze(0)
    # pytorch expects models in 4D shape. .unsqueeze() adds a batch dimension.

    with torch.no_grad():
        output = model(image)  # shape: [1, 10]
        probabilities = F.softmax(output, dim=1)  # convert logits to probabilities
        confidence, predicted_class = torch.max(probabilities, dim=1) # gets highest probability and class it corresponds to
        print(f"Predicted digit: {predicted_class.item()} with confidence {confidence.item() * 100:.2f}%")