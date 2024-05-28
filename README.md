# Speech Detection using TensorFlow

This repository contains a Jupyter Notebook for speech detection using TensorFlow. The project involves importing speech image data, preprocessing it, and training a machine learning model to detect speech patterns.

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/speech-detection.git
cd speech-detection
pip install numpy
pip install --upgrade pip
pip install tensorflow
# Optional: Install the preview build of TensorFlow (unstable)
pip install tf-nightly
```

## Data

The data used for this project is available from a GitHub repository. It consists of speech image data which is preprocessed and divided into training and validation sets.

To download and unzip the data, run the following commands:

```python
import tensorflow as tf

# Download and unzip the data
!wget https://raw.githubusercontent.com/andrsn/data/main/speechImageData.zip
!unzip -q speechImageData.zip
```

## Usage

The Jupyter Notebook provided in this repository guides you through the steps of importing data, preprocessing it, and training a machine learning model.

### Preprocess Data

The data is preprocessed into training and validation sets using Keras dataset objects:

```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='path/to/TrainData',
    labels='inferred',
    color_mode="grayscale",
    label_mode='categorical',
    batch_size=128,
    image_size=(98, 50)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory='path/to/ValData',
    labels='inferred',
    color_mode="grayscale",
    label_mode='categorical',
    batch_size=128,
    image_size=(98, 50)
)
```

### Model Training

The notebook includes detailed steps for training a TensorFlow model. Ensure you follow the steps outlined in the notebook to successfully train the model.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

1. Fork the repository.
2. Create a new branch: `git checkout -b my-feature-branch`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin my-feature-branch`.
5. Submit a pull request.

