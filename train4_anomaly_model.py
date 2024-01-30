import argparse
import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D, LayerNormalization
from keras.models import Sequential, load_model
#!pip install keras-layer-normalization

def model_fn(learning_rate, training_set):
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """
    
    #if not reload_model:
    #    return load_model(Config.MODEL_PATH,custom_objects={'LayerNormalization': LayerNormalization})
    #training_set = get_training_set()
    #training_set = np.array(training_set)
    print(training_set.shape)
    training_set = training_set.reshape(-1,10,256,256,1)
    seq = Sequential()
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, 256, 256, 1)))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
    print(seq.summary())
    seq.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-6 ))#decay=1e-5, ))
    
    return seq


def _load_training_data(base_dir):
    # Load your training data here
    # For example, load data from Numpy files
    X_train = np.load(os.path.join(base_dir, '0.npy'))
    for i in range(1, 10):
        X_tain = np.concatenate(X_train,    np.load(os.path.join(base_dir, str(i)+'.npy')))
    
    print(X_train.shape)
        
    
    #y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
    return X_train#, y_train

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
    print(os.listdir('/opt/'))
    print(os.listdir('/opt/ml/'))
    print(os.listdir('/opt/ml/input/'))
    print(os.listdir('/opt/ml/input/data/'))
    
    print(os.listdir('/opt/ml/input/data/train'))
    X_train = _load_training_data('/opt/ml/input/data/train')
    
    


    learning_rate = 0.01
    epochs = 1
    batch_size = 4
    # Create the model
    
    model = model_fn(args.learning_rate, X_train)


    # Train the model
    model.fit(X_train, X_train,
            batch_size=4, epochs=1, shuffle=False)


    # Evaluate the model
    #loss, accuracy = model.evaluate(X_test, y_test)

    #print(f'Test accuracy: {accuracy}')


    # Save the model
    model.save('/opt/ml/model/1')  # '1' is the model version
    #model.save(os.path.join(args.model_dir, '1'))  # '1' is the model version
