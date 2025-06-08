import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import argparse

# NN Architecture
class Digit_Classifier(nn.Module):
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def train_model(img, labels):
    opt.zero_grad() # resets gradient (which way to adjust weights, changes each epoch)
    model.train() # Set to Train Mode
    output = model(img) # Forward Pass
    loss = criterion(output, labels) # calc loss
    loss.backward() # backpropagate
    opt.step() # updatets model weights based on gradients computed during backpropagation
    return loss


def eval_model(img, labels):
    model.eval() # set to eval mode
    with torch.no_grad(): # do not store or compute gradients (since no training)
        output = model(img) # Forward Pass
        loss = criterion(output, labels) # loss
    return loss

if __name__ == '__main__':

    # Get Data + Transform to Tensor
    transforms_ops = transforms.Compose([transforms.ToTensor()])
    training_data = datasets.MNIST(root="./data", train=True, transform=transforms_ops, download=True)
    training_data_loader = torch.utils.data.DataLoader(training_data, shuffle=True, batch_size=1, num_workers=0)

    # add arguments for user to input
    parser = argparse.ArgumentParser(description="Train a digit classifier on MNIST")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--w', type=int, default=5, help='Eval Batch Number')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for Training')
    args = parser.parse_args()

    # Model + Loss Type + Optimizer
    model = Digit_Classifier()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # Args
    num_epochs = args.epochs
    batch_size = args.batch_size
    eval_batch_nr = args.w

    # Training Loop
    for epoch in range(num_epochs):
        for batch_nr, (img, labels) in enumerate(training_data_loader):
            if batch_nr != eval_batch_nr:  # train
                loss = train_model(img, labels)
                print(f"Epoch {epoch}: Batch {batch_nr}: Training Loss: {loss}")
            else:
                loss = eval_model(img, labels)
                print(f"Epoch {epoch}: Batch {batch_nr}: Eval Loss: {loss}")

            if batch_nr >= batch_size:
                break

    print(f"Training Done.")
    torch.save(model.state_dict(), "digit_classifier.pth")
    print("Model saved as digit_classifier.pth")


    # Define transform (should match your training transform)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Load test dataset
    test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    model.eval()  # set model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)  # forward pass
            _, predicted = torch.max(outputs.data, 1)  # get predicted class (index of max logit)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on test images: {accuracy:.2f}%")


