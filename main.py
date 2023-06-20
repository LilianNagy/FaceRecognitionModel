import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import random

dataset_path = '.\\images'  # Path to the directory containing the images
categories = ['jason', 'silvester', 'miley', 'cristiano', 'karim', 'jackie']  # List of categories

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the images and assign labels
images = []
labels = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    category_label = categories.index(category)

    for filename in os.listdir(category_path):
        image_path = os.path.join(category_path, filename)
        image = cv2.imread(image_path)

        if image is not None:
            # Convert image to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=5)

            # Skip images that do not have any detected faces
            if len(faces) == 0:
                continue

            for (x, y, w, h) in faces:
                # Extract the face region
                face_region = grayscale_image[y:y + h, x:x + w]

                # Resize the face region
                target_size = (128, 128)
                resized_face = cv2.resize(face_region, target_size)

                # Normalize the face region if desired
                normalized_face = resized_face / 255.0  # Normalize pixel values to the range [0, 1]

                images.append(normalized_face)
                labels.append(category_label)

# Convert the list of preprocessed images to numpy arrays
preprocessed_images = np.array(images)
labels = np.array(labels)

# Number of classes
num_classes = len(categories)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    preprocessed_images, labels, test_size=0.3, random_state=42, stratify=labels)

# Convert the labels to one-hot encoded vectors
train_labels = np.eye(num_classes)[train_labels]
test_labels = np.eye(num_classes)[test_labels]

# Architecture 1: Simple Feedforward Neural Network
model1 = Sequential()
model1.add(Flatten(input_shape=(128, 128)))  # Input layer, flatten the 128x128 images
model1.add(Dense(256, activation='relu'))  # Hidden layer with 256 neurons and ReLU activation
model1.add(Dense(128, activation='relu'))  # Hidden layer with 128 neurons and ReLU activation
model1.add(Dense(num_classes, activation='softmax'))  # Output layer

# Architecture 2: Convolutional Neural Network (CNN)
model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))  # Convolutional layer
model2.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
model2.add(Flatten())  # Flatten the feature maps
model2.add(Dense(128, activation='relu'))  # Hidden layer with 128 neurons and ReLU activation
model2.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the models
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the models
model1.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(test_images, test_labels))
model2.fit(train_images, train_labels, epochs=15, batch_size=32, validation_data=(test_images, test_labels))

# Evaluate the models
accuracy1 = model1.evaluate(test_images, test_labels)[1]
accuracy2 = model2.evaluate(test_images, test_labels)[1]

print("Accuracy of Model 1:", accuracy1)
print("Accuracy of Model 2:", accuracy2)


# Define the mapping of class labels to names
label_names = {
    0: 'Jason Statham',
    1: 'Sylvester Stallone',
    2: 'Miley Cyrus',
    3: 'Cristiano',
    4: 'Karim Abdelaziz',
    5: 'Jackie Chan'
}

# Show 10 random results from the testing set
num_samples = 10
random_indices = random.sample(range(len(test_images)), num_samples)

print("Random Results from the Testing Set:")
for idx in random_indices:
    sample_image = test_images[idx]
    sample_label = test_labels[idx]

    # Reshape the sample image for prediction
    sample_image = sample_image.reshape(1, 128, 128)

    # Predict using both models
    prediction1 = model1.predict(sample_image)[0]
    prediction2 = model2.predict(sample_image)[0]

    # Get the predicted class labels
    predicted_class1 = np.argmax(prediction1)
    predicted_class2 = np.argmax(prediction2)

    # Get the true class label
    true_class = np.argmax(sample_label)

    print("Sample:", idx)
    print("True Class:", label_names[true_class])
    print("Model 1 Prediction:", label_names[predicted_class1])
    print("Model 2 Prediction:", label_names[predicted_class2])
    print()
