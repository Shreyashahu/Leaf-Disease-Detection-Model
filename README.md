# Basic-ML-Model

This repository contains a basic machine learning model using TensorFlow to classify images into two categories: Healthy and Diseased.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Testing](#testing)
- [Saving the Model](#saving-the-model)
- [Usage](#usage)

## Overview

This project involves building a Convolutional Neural Network (CNN) model to classify images of plants into two categories: healthy and diseased. The model is built using TensorFlow and Keras.

## Dependencies

This project requires the following Python libraries:
- TensorFlow
- Keras
- Matplotlib

To install the dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project consists of images organized in the following directory structure:

```
dataset/
  ├── train/
  │   ├── healthy/
  │   └── diseased/
  ├── validation/
  │   ├── healthy/
  │   └── diseased/
  └── test/
      ├── healthy/
      └── diseased/
```

- **train/**: Contains the training images.
- **validation/**: Contains the validation images.
- **test/**: Contains the test images.

## Model

The model is a Convolutional Neural Network (CNN) built using TensorFlow/Keras. It uses the following layers:
- Convolutional layers for feature extraction.
- Max-pooling layers to reduce dimensionality.
- Dense layers for classification.

The model is trained using the `Adam` optimizer and `SparseCategoricalCrossentropy` as the loss function.

## Training

To train the model, run the following script:

```bash
python hello.py
```

The script will:
- Load the dataset from the specified directories.
- Preprocess the data (resize images and normalize).
- Train the model on the training data.
- Validate the model on the validation data.

## Testing

After training, the model can be evaluated on the test dataset to check its accuracy:

```bash
python hello.py
```

This will print the test accuracy and loss.

## Saving the Model

Once the model is trained, it is saved in the Keras format as a `.keras` file. To load and use the model later, you can use the following code:

```python
from tensorflow.keras.models import load_model

# Load the model
model = load_model('my_model.keras')

# Use the model to make predictions
predictions = model.predict(test_data)
```

## Usage

After training and saving the model, you can use it to make predictions on new images. The following code can be used to load the model and make predictions:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('my_model.keras')

# Load an image to predict
img = image.load_img('path_to_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make a prediction
prediction = model.predict(img_array)

# Print the prediction (healthy or diseased)
if prediction[0][0] > 0.5:
    print("Healthy")
else:
    print("Diseased")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This version reflects the use of the `.keras` format for saving the model. Let me know if you need any more adjustments!
