# Number Image Detection ML 🔢 👁️

A Machine Learning computer vision project designed to detect, classify, and recognize handwritten or digital numbers from raw image data using a Convolutional Neural Network (CNN).

---

## Overview

This repository contains a PyTorch-based deep learning pipeline for image-based digit recognition. The model is trained on the MNIST dataset to classify digits from 0 to 9. It extracts spatial features using convolutional and pooling layers, and outputs a classification prediction with a calculated confidence percentage.

---

## Project Structure

The project consists of two primary scripts:

* **`train_model.py`**: The training script. It defines the `Digit_Classifier` CNN architecture (featuring `Conv2d`, `ReLU`, `MaxPool2d`, and `Linear` layers). It automatically downloads the MNIST dataset, trains the model using the Adam optimizer and CrossEntropyLoss, evaluates its accuracy on the test set, and saves the trained weights as `digit_classifier.pth`.
* **`test_model.py`**: The inference script. It loads the saved `digit_classifier.pth` model, reads a raw image file provided via the command line, and applies necessary transformations (converting to grayscale, resizing to 28x28, and normalizing) to match the MNIST training data. It then uses a softmax function to print the predicted digit along with its confidence score.

---

## Tech Stack & Dependencies

* **Language:** Python 3.9+
* **Deep Learning:** PyTorch (`torch`, `torch.nn`), Torchvision (`datasets`, `transforms`)
* **Image Processing:** PIL (Pillow)

---

## Local Setup & Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/mhach06/number_img_detection_ML.git](https://github.com/mhach06/number_img_detection_ML.git)
cd number_img_detection_ML
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision pillow argparse

```

---

## How to Train the Model

The `train_model.py` script comes with built-in command-line arguments so you can easily adjust the training hyperparameters.

Run the training loop:

```bash
python train_model.py --epochs 10 --batch_size 64

```

**Available Arguments:**

* `--epochs`: Number of training epochs (default is 10).
* `--batch_size`: Batch size for training (default is 64).
* `--w`: The batch number to evaluate on during the training loop (default is 5).

Upon completion, the script will calculate the overall test accuracy and save the trained model file as `digit_classifier.pth` directly into the project root.

---

## How to Test the Model (Inference)

To run predictions on a new, unseen image:

Execute the script from the command line, passing the path to the image you want to test.

```bash
python test_model.py --image path/to/your/image.png

```

**Available Arguments:**

* `--image` (Required): The file path to the image you want to test.
* `--model` (Optional): The file path to the saved model weights. Defaults to `digit_classifier.pth` in the current directory.

*The script will process the image, feed it through the CNN, and print the predicted number (0-9) and the confidence percentage to the console.*
