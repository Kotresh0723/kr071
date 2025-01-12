This repository contains a Python-based face detection project leveraging a pre-trained model from Google Teachable Machine. It offers an easy-to-implement solution for detecting faces in images and video streams using machine learning techniques.

# Features
- Detects faces in images and in real-time too.
- Utilizes a pre-trained model from Google Teachable Machine.
- Beginner-friendly and easy to customize.

# Prerequisites
- Python 3.7 or higher
- OpenCV library
- NumPy library

-Installlation of facefinder+ 
-Step1: Open chrome and search for Google Teachable Machines website(https://teachablemachine.withgoogle.com/) and click on Get started. 
![2025-01-09](https://github.com/user-attachments/assets/a47946d7-b21a-4719-a0a2-58767be39888)

-Step2: And under the New Project section, select the Image Model folder.
![2025-01-10](https://github.com/user-attachments/assets/cb2c3ee2-b959-4149-b727-b88df79302c7)

-Step3: Select the Standard Image Model option.
![2025-01-09 (1)](https://github.com/user-attachments/assets/a43bc4bd-02d6-4266-ba97-3f65329c9fde)

-Step4: Determine the Required classes and upload the photos using webcam and google drive, And Train the Model. You could see the comparision percentage for the given classes.
![2025-01-10 (1)](https://github.com/user-attachments/assets/c1dd753c-cce3-497c-8c26-bb44d02b5750)

-Step5: Click on Export the Model.
![2025-01-10 (4)](https://github.com/user-attachments/assets/8f025611-6f8c-4228-9557-7c971ac79f90)

-Step6: Under the tensorflow tab, select Download the model option. And a .zip file would be downloaded. 
![2025-01-10 (6)](https://github.com/user-attachments/assets/c3da9765-1744-4e80-b75b-c9291de9ed6f)

-Step8: To run it in local system, paste the Given code in pycharm or any other code Interpreter.
![2025-01-10 (10)](https://github.com/user-attachments/assets/9e78cf89-5b15-4c65-bf54-3ab234e459f5)

-Step9: Run the code and, Hence the facefinder+ works.

Step10: create a python file in your pycharm IDE as (main.py)

from tensorflow.keras.models import load_model
# TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels, stripping any newline characters
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)




while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name, end="")  # Removed slicing to print the full class name
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

 Step11: Run the code output will be expected with the confidence score of your model in pycharm. It will directly connect to Your Webcam directly and give you the output

 Step12:Expected errors 
