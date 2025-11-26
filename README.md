# Project Based Experiments
## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow:
## Dataset Acquisition:
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing:
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting:
Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model:
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training:
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation:
Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning:
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization:
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

# Program:
```py
# mnist_mlp_keras.py
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import pickle, os

# 1) Load MNIST
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

# 2) Preprocess
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train_full_flat = x_train_full.reshape((x_train_full.shape[0], -1))
x_test_flat = x_test.reshape((x_test.shape[0], -1))
y_train_full_cat = to_categorical(y_train_full, 10)
y_test_cat = to_categorical(y_test, 10)

# 3) Split train -> train + val
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full_flat, y_train_full_cat, test_size=0.1, random_state=42, stratify=y_train_full
)

# 4) Build MLP
def build_mlp(input_dim, hidden_layers=(512,256), dropout_rate=0.2, lr=1e-3):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_mlp(x_train.shape[1], hidden_layers=(512,256), dropout_rate=0.2, lr=1e-3)
model.summary()

# 5) Train
out_dir = "saved_models"
os.makedirs(out_dir, exist_ok=True)
checkpoint_path = os.path.join(out_dir, "mnist_mlp_best.h5")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True)
]
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=50, batch_size=128, callbacks=callbacks, verbose=2)

# 6) Evaluate
test_preds_prob = model.predict(x_test_flat)
test_preds = np.argmax(test_preds_prob, axis=1)
test_acc = accuracy_score(y_test, test_preds)
print(f"Test accuracy: {test_acc:.4f}")

cm = confusion_matrix(y_test, test_preds)
print("Confusion matrix:\n", cm)
print("Classification report:\n", classification_report(y_test, test_preds))

# 7) Plots
epochs = range(1, len(history.history['loss'])+1)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(epochs, history.history['loss'], label='train loss')
plt.plot(epochs, history.history['val_loss'], label='val loss')
plt.legend(); plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(epochs, history.history['accuracy'], label='train acc')
plt.plot(epochs, history.history['val_accuracy'], label='val acc')
plt.legend(); plt.title('Accuracy')
plt.show()

# 8) Misclassified samples
x_test_images = x_test  # 28x28 images
mis_idx = np.where(test_preds != y_test)[0]
print("Misclassified count:", len(mis_idx))
for i, idx in enumerate(mis_idx[:12]):
    plt.subplot(2,6,i+1); plt.imshow(x_test_images[idx], cmap='gray'); plt.title(f"T:{y_test[idx]} P:{test_preds[idx]}"); plt.axis('off')
plt.show()

# 9) Save final model and history
model_save_path = os.path.join(out_dir, "mnist_mlp_model.h5")
model.save(model_save_path)
with open(os.path.join(out_dir,"history.pkl"), "wb") as f:
    pickle.dump(history.history, f)
print("Saved model to:", model_save_path)
```

## Output:
<img width="831" height="353" alt="image" src="https://github.com/user-attachments/assets/874994ed-5336-4f1c-bdb8-422038ec9ddf" />

<img width="533" height="638" alt="image" src="https://github.com/user-attachments/assets/edd65274-08d2-4816-a97c-67e735cf999d" />

<img width="1310" height="472" alt="image" src="https://github.com/user-attachments/assets/24a9e352-2fa6-44ab-bbc6-45fe6eec451f" />

<img width="691" height="417" alt="image" src="https://github.com/user-attachments/assets/279b4d7a-404b-4d0c-9f88-e6a14db2b2b5" />
