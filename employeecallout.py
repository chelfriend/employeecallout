import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Example dataset (replace this with your own dataset)
# Here, 'X' represents the features (e.g., reasons for calling out) and 'y' represents the target variable (number of employees on leave)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 10, size=(100, 4))  # 100 samples, 4 Fridays in a month

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='linear')  # Output layer with linear activation
])

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Mean Squared Error loss function

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Make predictions
predictions = model.predict(X_test)
print("Sample predictions:", predictions[:5])
