# Diabetes Prediction Using Machine Learning

## Project Overview

This project aims to predict the likelihood of diabetes in individuals using the Pima Indians Diabetes Dataset. The dataset contains various health-related attributes, and the goal is to build a machine learning model that can accurately predict the outcome based on these features.

## Problem Statement

Diabetes is a chronic medical condition that occurs when the body is unable to properly process glucose. Early detection of diabetes can lead to better management and treatment, improving patient outcomes. The goal of this project is to develop a machine learning model capable of predicting whether a person has diabetes based on their health indicators.

## Libraries and Frameworks Used

`NumPy (Version: 1.22.3)`

`Pandas (Version: 2.0.3)`

`Matplotlib (Version: 3.7.2)`

`Theano (Version: 1.0.5)`

`TensorFlow (Version: 2.10.0)`

`Keras (Version: 2.10.0)`

`Scikit-learn (Version: 1.3.0)`

`Seaborn (Version: 0.13.2)`

These libraries were used to perform data preprocessing, model training, and evaluation, providing an efficient way to handle and manipulate the data and implement machine learning algorithms.

## Dataset

The project uses the Pima Indians Diabetes Dataset, which consists of 768 instances with 8 feature variables:

1. Pregnancies - Number of times pregnant
2. Glucose - Plasma glucose concentration
3. BloodPressure - Diastolic blood pressure (mm Hg)
4. SkinThickness - Triceps skinfold thickness (mm)
5. Insulin - 2-Hour serum insulin (mu U/ml)
6. BMI - Body mass index (weight in kg / height in m^2)
7. DiabetesPedigreeFunction - A function that scores the likelihood of diabetes based on family history
8. Age - Age of the individual

The target variable (Outcome) indicates whether the individual has diabetes (1) or not (0).

## Demo
<img src="https://github.com/saikumarpeeka/Diabetes-Prediction/blob/main/Project%20Files%2FDiabetes-Prediction.gif" width="300">

# Model Implementation

## Data Preprocessing

1.Label Encoding: Applied to categorical columns.

2.Feature Scaling: Used StandardScaler to normalize the features for training.

3.Training and Testing Split: Divided the data into 80% training and 20% testing using train_test_split.

## Model Architecture

A Fully Connected Neural Network (ANN) was used to classify the data:

Input Layer: 6 neurons with ReLU activation.

Hidden Layer: 1 hidden layer with 6 neurons, ReLU activation.

Output Layer: 1 neuron with Sigmoid activation for binary classification.

## Model Compilation

Optimizer: Adam optimizer
Loss Function: Binary crossentropy (for binary classification)
Metric: Accuracy
The model was trained for 100 epochs using the training data.

# Results

After training for 100 epochs, the model achieved an accuracy rate of 98.86%, indicating its strong performance in predicting diabetes based on the provided features.

# Installation & Setup

### 1️⃣ Install Anaconda

Download and install Anaconda <a href="https://www.anaconda.com/download">Here</a>

### 2️⃣ Open Anaconda Prompt

#### Launch the Anaconda Prompt and follow the steps below.

### 3️⃣ Install Jupyter Notebook

#### If not already installed, run:

```sh
conda install jupyter
```

### 4️⃣ Install Theano

```sh
conda install theano_env
```

#### Activate Theano:

```sh
conda activate theano_env
```

### 5️⃣ Install Required Libraries

#### Install each required library separately:

```sh
pip install pandas

pip install keras

pip install tensorflow

pip install matplotlib

pip install scikit-learn

pip install seaborn
```

### 6️⃣ Run Jupyter Notebook

#### Start Jupyter Notebook by executing:

```sh
jupyter notebook
```

#### Open the saved Diabetes Prediction notebook and execute the code cells.

# License

### This project is licensed under the MIT License.

# Acknowledgment

### Pima Indians Diabetes Dataset: Source dataset used for training and evaluation.

### TensorFlow & Keras: Machine learning frameworks used for building and training the neural network model.

# Errors and Warnings

## Warning-1

##### You may encounter the following warning:

`WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.`

This is not an error but a warning indicating that Theano is using a NumPy-based implementation for BLAS (Basic Linear Algebra Subprograms) functions. This may slightly affect performance.

The code will run successfully, but for better performance, you can consider optimizing the BLAS implementation.

## Warning-2

##### While running the code, you may encounter the following warning:

`C:\Users\Pavan Kalyan\anaconda3\envs\theano_env\lib\site-packages\seaborn\axisgrid.py:123: UserWarning: The figure layout has changed to tight self._figure.tight_layout(*args, **kwargs)`

Explanation: This warning is related to Seaborn's behavior when adjusting the layout of figures for better presentation. It occurs when Seaborn automatically adjusts the layout of a plot to ensure that labels and titles fit within the figure area.

Action: This is not an error, and the code will continue to run successfully. It is a typical warning from Seaborn to inform you that the figure layout has been automatically adjusted to be tighter.

If you want to suppress this warning, you can modify the plot settings, but it doesn't affect the functionality of the code.
