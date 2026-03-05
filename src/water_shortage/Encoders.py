import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set random seed for reproducibility
tf.random.set_seed(42)

# Example data generation
# Replace this with your actual dataset
X = np.random.rand(1000, 20)  # 1000 samples, 20 features
y = np.random.randint(0, 5, size=(1000, 1))  # 5 classes

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ohe = OneHotEncoder(sparse=False)
y_onehot = ohe.fit_transform(y)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Define the autoencoder
input_dim = X_train.shape[1]
encoding_dim = 10  # Dimensionality of the encoding space

# Autoencoder architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Build the autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(
    X_train, X_train,  # Autoencoders learn to reconstruct input data
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

# Extract encoder model for feature extraction
encoder = Model(inputs=input_layer, outputs=encoded)

# Use the encoder to transform the data
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Build a classification model on top of the encoded features
classifier = Sequential([
    Dense(64, activation='relu', input_dim=encoding_dim),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')  # Output layer for classification
])

# Compile the classifier
classifier.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the classifier
classifier.fit(
    X_train_encoded, y_train,
    validation_data=(X_test_encoded, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the classifier
loss, accuracy = classifier.evaluate(X_test_encoded, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")

y_test.to_csv("D:/HiParis_Hackathon/y_test_Encoder.csv", index = False)
