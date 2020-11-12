import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets

# LOAD MNIST SET
training_data_path = "../../dataset/"

training_x = np.load(training_data_path + 'MNIST-training-samples.npy')
image_width = training_x.shape[1]
image_height = training_x.shape[2]

training_x = training_x.astype('float')/255
training_x = training_x.reshape(-1, training_x.shape[1]*training_x.shape[2])

testing_x = np.load(training_data_path + 'MNIST-testing-samples.npy')
testing_x = testing_x.astype('float')/255
testing_x = testing_x.reshape(-1, testing_x.shape[1]*testing_x.shape[2])

training_y = np.load(training_data_path + 'MNIST-training-labels.npy')
testing_y = np.load(training_data_path + 'MNIST-testing-labels.npy')

# training_x = training_x[:50000]
# training_y = training_y[:50000]

# LOAD TRAINING RESULTS
training_data_path = "../../results/CRBA-2000/"
labeling_rates = np.load(training_data_path + "labeling_rates.npy")
testing_rates = np.load(training_data_path + "testing_rates.npy")


# SYSTEM PARAMETERS
number_neurons = labeling_rates.shape[0]
number_training_samples = training_x.shape[0]
number_testing_samples = testing_x.shape[0]

print number_neurons
print number_training_samples
print number_testing_samples


# TRAINING NEURAL NETWORK

# Create add on neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2000, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# FIT MODEL
batch_size = 10
epochs = 50

number_winners = 1500

# process training features
labeling_rank = np.argsort(labeling_rates, axis=0)[-1:-number_winners-1:-1, :]
print labeling_rank.shape

labeling_winner = np.zeros((number_neurons, number_training_samples))
for i in range(number_training_samples):
    labeling_winner[:, i][labeling_rank[:, i]] = labeling_rates[:, i][labeling_rank[:, i]]

training_samples = labeling_winner.transpose()
training_labels = training_y.astype('float').reshape(number_training_samples, -1)

# process testing features
testing_rank = np.argsort(testing_rates, axis=0)[-1:-number_winners-1:-1, :]

testing_winner = np.zeros((number_neurons, number_testing_samples))
for i in range(number_testing_samples):
    testing_winner[:, i][testing_rank[:, i]] = testing_rates[:, i][testing_rank[:, i]]

validating_samples = testing_winner.transpose()
validating_labels = testing_y.astype('float').reshape(number_testing_samples, -1)

# training_samples = training_x
# validating_samples = testing_x

# train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x=training_samples, y=training_labels, batch_size=batch_size, epochs=epochs, verbose=1,
                 validation_data=(validating_samples, validating_labels))

val_hist = np.array(hist.history['val_acc'])
max_val = val_hist.max()

# print validation
print val_hist
print max_val