import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import glob


# Set random seed for reproducibility
tf.random.set_seed(42)

def build_dnn_model(input_shape, num_classes=5):
    """
    Builds a deep neural network model for classification.
    
    Parameters:
    - input_shape (int): Number of features in the input data.
    - num_classes (int): Number of output classes.
    
    Returns:
    - model: Compiled Keras model.
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(512, input_dim=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Hidden layers
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Example usage
if __name__ == "__main__":
    # Assuming you have a dataset with features `X` and labels `y`
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    

    X = np.load("D:/HiParis_Hackathon/hickathon/X_for_DNN.npy")
    y = np.load("D:/HiParis_Hackathon/hickathon/y_for_DNN.npy")

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build and train the model
    input_shape = X_train.shape[1]
    model = build_dnn_model(input_shape, num_classes=5)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.2f}")

    y_test.to_csv("D:/HiParis_Hackathon/y_test_DNN.csv")