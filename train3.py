import argparse
import os
import numpy as np
import tensorflow as tf

def model_fn(learning_rate):
    # Create a Keras model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),  # First hidden layer
        tf.keras.layers.Dropout(0.3),  # Dropout layer to reduce overfitting
        tf.keras.layers.Dense(32, activation='relu'),  # Second hidden layer
        tf.keras.layers.Dense(3, activation='softmax')  # Output layer with softmax for 3 classes
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),  # Use CategoricalCrossentropy
                  metrics=['accuracy'])
    return model

def _load_training_data(base_dir):
    # Load your training data here
    # For example, load data from Numpy files
    X_train = np.load(os.path.join(base_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
    return X_train, y_train

def _load_testing_data(base_dir):
    # Load your testing data here
    X_test = np.load(os.path.join(base_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(base_dir, 'y_test.npy'))
    return X_test, y_test

def _parse_args():
    parser = argparse.ArgumentParser()
    # Add argument definitions (e.g., learning rate, epochs)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_dir', type=str)  # SageMaker will provide this argument
    return parser.parse_known_args()

if __name__ == '__main__':
    
    args, _ = _parse_args()

    # Load data
    X_train, y_train = _load_training_data('/opt/ml/input/data/train')

    X_test, y_test = _load_testing_data('/opt/ml/input/data/test')


    learning_rate = 0.001
    epochs = 50
    batch_size = 32
    # Create the model
    
    model = model_fn(args.learning_rate)


    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    print(f'Test accuracy: {accuracy}')


    # Save the model
    model.save('/opt/ml/model/1')  # '1' is the model version
    #model.save(os.path.join(args.model_dir, '1'))  # '1' is the model version
