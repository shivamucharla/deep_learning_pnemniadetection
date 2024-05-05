Pneumonia Detection using Transfer Learning (Xception)

Introduction:
This repository contains code for a pneumonia detection system developed using transfer learning with the Xception architecture, a pre-trained deep learning model. Pneumonia is a severe respiratory condition that requires early detection for effective treatment. The goal of this project is to leverage transfer learning to improve the performance of the model on a smaller dataset specific to pneumonia detection.

Files:

pneumonia_detection.ipynb: Jupyter Notebook containing the code for training the pneumonia detection model. This notebook covers data preprocessing, model building, training, evaluation, and saving the trained model.
pneumonia_detection_app.py: Streamlit application file for deploying the trained model. This file includes the user interface and functionality to upload chest X-ray images and predict whether the person is affected by pneumonia or not. The accuracy of the model (94.03%) is displayed to the user.
data/: Directory containing the dataset used for training the model. This dataset consists of chest X-ray images categorized into pneumonia and normal classes.

Additional Information:

EDA (Exploratory Data Analysis): Before training the model, exploratory data analysis was conducted to understand the distribution of classes, visualize sample images, and identify any data preprocessing requirements.

Transfer Learning with Xception: Transfer learning involves utilizing a pre-trained neural network model (Xception) and fine-tuning it on our specific dataset for pneumonia detection. This approach helps achieve higher accuracy with less training data and computational resources.

Instructions for Running the Streamlit Application:

1.Install the necessary dependencies by running pip install -r requirements.txt.

2.Navigate to the directory containing pneumonia_detection_app.py.

3.Run the Streamlit application using the command streamlit run pneumonia_detection_app.py.

4.Once the application is running, upload a chest X-ray image using the provided interface.

5.The application will process the image and display the prediction result, indicating whether the person is affected by pneumonia or not.
6.Additionally, the accuracy of the model (94.03%) will be shown to the user.
