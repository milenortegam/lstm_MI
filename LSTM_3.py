import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the CSV data
data = pd.read_csv('exp4/concatenated_exp4.csv')


data = data.drop(['frame'], axis=1)

# Encode labels
label_encoder = LabelEncoder()
labels = data['class'].values
label_encoder.fit(labels)
data['class'] = label_encoder.transform(labels)

# Get unique face IDs
unique_face_ids = data['face_id'].unique()

# Create lists to store the reshaped data and labels
reshaped_data = []
labels = []

# Iterate over unique face IDs
for face_id in unique_face_ids:
    face_data = data[data['face_id'] == face_id].drop(['face_id'], axis=1).values
    label = data[data['face_id'] == face_id]['class'].iloc[0]  # Assume the labels are the same for all frames of a face_id
    reshaped_data.append(face_data)
    labels.append(label)

# Pad sequences to ensure equal length
padded_data = tf.keras.preprocessing.sequence.pad_sequences(reshaped_data, padding='post')

# Convert lists to NumPy arrays
X = np.array(padded_data)
y = np.array(labels)

# Encode labels
num_classes = len(label_encoder.classes_)
y_one_hot = to_categorical(y, num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
print("Test loss:", loss)
print("Test accuracy:", accuracy)
