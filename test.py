import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model from the file
model = load_model('covid19_vgg19_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Path to the directory containing the images you want to predict
images_dir = 'Data/Test'

# List all files in the directory
image_files = os.listdir(images_dir)

# Calculate the number of rows and columns for subplots
num_images = len(image_files)
num_rows = (num_images + 3) // 4  # Round up to the nearest multiple of 4
num_cols = min(4, num_images)  # Maximum 4 columns

# Initialize subplot
plt.figure(figsize=(15, 10))

# Iterate over each image file
for i, filename in enumerate(image_files, start=1):
    # Check if the file is an image file (JPEG, PNG, etc.)
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Load and process the image
        img_path = os.path.join(images_dir, filename)
        img = image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the data

        # Predict or classify the image
        predictions = model.predict(img_array)

        # Check the prediction result
        if predictions[0][0] < 0.5:
            prediction_label = "Covid-19"
        else:
            prediction_label = "Không phải Covid-19"

        # Add subplot
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(img_array[0, :, :, 0], cmap='gray')
        plt.title(f"{filename}: {prediction_label}")
        plt.axis('off')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
