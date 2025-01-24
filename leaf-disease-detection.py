# Step 1: Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import image_dataset_from_directory

# Step 2: Define paths to your dataset directories
base_dir = 'C:\\Users\\91898\\OneDrive\\Desktop\\Pleasee\\dataset'  # Replace with your dataset path
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Step 3: Load datasets
train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),  # Resize images
    batch_size=32
)

val_dataset = image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32
)

# Step 4: Normalize the dataset by rescaling the pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Step 5: Build the model
model = models.Sequential([
    # Input layer
    layers.Input(shape=(224, 224, 3)),  # Define the input shape
    
    # First convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the output to feed into dense layers
    layers.Flatten(),
    
    # Dense layers
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 classes: Healthy and Diseased
])

# Step 6: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])

# Step 7: Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10  # You can adjust the number of epochs based on your needs
)

# Step 8: Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# Visualizing the accuracy and loss over epochs
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

model.save('C:\\Users\\91898\\OneDrive\\Desktop\\Models\\leaf-disease-detection.keras')
