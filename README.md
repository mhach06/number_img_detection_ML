# Number Image Detection ML 🔢 👁️

A Machine Learning computer vision project designed to detect, classify, and recognize handwritten or digital numbers from raw image data using a Convolutional Neural Network (CNN)[cite: 2].

---

## Overview

This repository contains a PyTorch-based deep learning pipeline for image-based digit recognition. The model is trained on the MNIST dataset to classify digits from 0 to 9[cite: 2]. It extracts spatial features using convolutional and pooling layers, and outputs a classification prediction with a calculated confidence percentage[cite: 1, 2].

---

## Project Structure

The project consists of two primary scripts:

* **`train_model.py`**: The training script. It defines the `Digit_Classifier` CNN architecture (featuring `Conv2d`, `ReLU`, `MaxPool2d`, and `Linear` layers)[cite: 2]. It automatically downloads the MNIST dataset, trains the model using the Adam optimizer and CrossEntropyLoss, evaluates its accuracy on the test set, and saves the trained weights as `digit_classifier.pth`[cite: 2].
* **`test_model.py`**: The inference script. It loads the saved `digit_classifier.pth` model, reads a raw image file, and applies necessary transformations (converting to grayscale, resizing to 28x28, and normalizing) to match the MNIST training data[cite: 1]. It then uses a softmax function to print the predicted digit along with its confidence score[cite: 1].

---

## Tech Stack & Dependencies

* **Language:** Python 3.9+
* **Deep Learning:** PyTorch (`torch`, `torch.nn`), Torchvision (`datasets`, `transforms`)[cite: 1, 2]
* **Image Processing:** PIL (Pillow)[cite: 1]

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

1. Open `test_model.py` and ensure the `img_path` variable points to your desired test image (e.g., `unnamed.png`).


2. Ensure the `torch.load()` path points to your generated `digit_classifier.pth` file.


3. Execute the script:

```bash
python test_model.py

```

The script will process the image, feed it through the CNN, and print the predicted number (0-9) and the confidence percentage to the console.

```

```