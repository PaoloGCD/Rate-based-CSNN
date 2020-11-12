'''
CRBA
Competitive Rate-Based Algorithm Based on Competitive Spiking Neural Networks

@author: Paolo G. Cachi
'''

import numpy as np
import xmltodict

import cv2
import time
import sys


# SYSTEM PARAMETERS
parameters = "../../params/CRBA-2000.xml"
result_path = "../../results/CRBA-2000/"
if len(sys.argv) == 3:
    parameters = str(sys.argv[1])
    result_path = str(sys.argv[2])

with open(parameters) as fd:
    params = xmltodict.parse(fd.read())

spiking_neurons = int(params['params']['network']['spiking_neurons'])
running_time = int(params['params']['network']['running_time'])

alpha_number_spikes = float(params['params']['network']['alpha_number_spikes'])
alpha_weight_update = float(params['params']['network']['alpha_weight_update'])

threshold_0 = float(params['params']['network']['threshold_0'])
threshold_offset = float(params['params']['network']['threshold_offset'])
threshold_update = float(params['params']['network']['threshold_update'])
threshold_tau = float(params['params']['network']['threshold_tau'])

weight_blurring_number = int(params['params']['network']['weight_initialization_blurring'])
weight_total_connection = float(params['params']['network']['weight_total_value'])

training_number_epochs = int(params['params']['training']['number_epochs'])
training_print_step = int(params['params']['training']['print_step'])
training_save_step = int(params['params']['training']['save_step'])
validation_step = int(params['params']['training']['validation_step'])
test_step = int(params['params']['training']['test_step'])

validation_number_samples = int(params['params']['validation']['validation_samples'])
validation_window = int(params['params']['validation']['validation_window'])

training_data_x_path = str(params['params']['dataset']['training_x']['@path'])
training_data_y_path = str(params['params']['dataset']['training_y']['@path'])
testing_data_x_path = str(params['params']['dataset']['testing_x']['@path'])
testing_data_y_path = str(params['params']['dataset']['testing_y']['@path'])
number_classes = int(params['params']['dataset']['number_classes'])

print 'EXPERIMENT', result_path


# LOAD MNIST SET
training_set_x = np.load(training_data_x_path)
image_width = training_set_x.shape[1]
image_height = training_set_x.shape[2]

training_set_x = training_set_x.astype('float')/255
training_set_x = training_set_x.reshape(-1, training_set_x.shape[1]*training_set_x.shape[2])

training_set_y = np.load(training_data_y_path)

testing_set_x = np.load(testing_data_x_path)
testing_set_x = testing_set_x.astype('float')/255
testing_set_x = testing_set_x.reshape(-1, testing_set_x.shape[1]*testing_set_x.shape[2])

testing_set_y = np.load(testing_data_y_path)

training_x = training_set_x[:-validation_number_samples, :]
training_y = training_set_y[:-validation_number_samples]

validation_x = training_set_x[-validation_number_samples:, :]
validation_y = training_set_y[-validation_number_samples:]

testing_x = testing_set_x
testing_y = testing_set_y


# SAVE PARAMETERS
description = '''
DiDIRECT CSNN:

spiking_neurons = %d
blurring_number = %d
number_validation_samples = %d

number_epochs = %d
running_time = %d
alpha_number_spikes = %f
alpha_weight_update = %f

threshold_0 = %d
threshold_offset = %d
threshold_update = %f
threshold_tau = %f

validation_step = %s
validation_labeling_window = %d

save_weights = %i
''' % (spiking_neurons, weight_blurring_number, validation_x.shape[0], training_number_epochs, running_time,
       alpha_number_spikes, alpha_weight_update, threshold_0, threshold_offset, threshold_update, threshold_tau,
       validation_step.__str__(), validation_window, training_save_step)
description_file = open(result_path + "description.txt", "w+")
description_file.write(description)
description_file.close()


# INPUT SAMPLE INITIALIZATION
number_training_samples = training_x.shape[0]
total_training_samples = training_number_epochs * number_training_samples

sensory_neurons = training_x.shape[1]
input_indexes = np.zeros(total_training_samples).astype("int")
for i in range(training_number_epochs):
    rand_index = np.arange(number_training_samples)
    np.random.shuffle(rand_index)
    input_indexes[i * number_training_samples:(i + 1) * number_training_samples] = rand_index.astype("int")
    # input_indexes[i * number_training_samples:(i + 1) * number_training_samples] = np.arange(number_training_samples)


# WEIGHT INITIALIZATION
initial_weights = np.copy(training_x[input_indexes[-spiking_neurons:]])

# apply blurring
if weight_blurring_number > 0:
    for i in range(spiking_neurons):
        blur = cv2.blur(initial_weights[i].reshape(image_width, image_height), (weight_blurring_number, weight_blurring_number))
        initial_weights[i] = blur.reshape(1, sensory_neurons)

# normalize
total_input_weight_neuron = np.sum(initial_weights, axis=1).reshape(spiking_neurons, 1)
initial_weights *= weight_total_connection / total_input_weight_neuron

default_weight = np.copy(initial_weights)
weights = initial_weights
np.save(result_path + "partial_weight_%06d" % 0, weights)


# THRESHOLD INITIALIZATION
threshold = np.ones(spiking_neurons) * threshold_0
np.save(result_path + "partial_threshold_%06d" % 0, threshold)


# TRAINING PROCESS
winner_neuron = 0
training_rates = np.zeros((total_training_samples, spiking_neurons))
validation_accuracies = []
testing_accuracies = []

training_time_start = time.time()
loop_time_start = time.time()

if validation_step > 0:

    # Labeling
    labeling_x = training_x[:validation_window]
    labeling_y = training_y[:validation_window]

    labeling_rates = np.dot(weights, labeling_x.transpose()) / threshold.reshape((spiking_neurons, 1))
    labeling_rank = np.argmax(labeling_rates, axis=0)

    labeling_winner = np.zeros((spiking_neurons, validation_window))
    for j in range(validation_window):
        labeling_winner[labeling_rank[j], j] = labeling_rates[labeling_rank[j], j]

    neuron_class_rate = np.zeros((spiking_neurons, number_classes))
    for j in range(number_classes):
        class_rate = labeling_winner[:, (labeling_y == j).reshape(validation_window)]
        neuron_class_rate[:, j] = class_rate.sum(axis=1)

    neuron_labels = np.argmax(neuron_class_rate, axis=1)

    # Validation
    validation_rates = np.dot(weights, validation_x.transpose()) / threshold.reshape((spiking_neurons, 1))
    validation_rank = np.argmax(validation_rates, axis=0)

    validation_labels = neuron_labels[validation_rank]

    validation_labels_difference = validation_y.reshape(-1) - validation_labels
    validation_labels_correct = len(np.where(validation_labels_difference == 0)[0])
    validation_accuracy = validation_labels_correct / float(validation_x.shape[0])

    validation_accuracies.append([0, validation_accuracy])

if test_step > 0:

    # Labeling
    labeling_x = training_x
    labeling_y = training_y

    labeling_rates = np.dot(weights, labeling_x.transpose()) / threshold.reshape((spiking_neurons, 1))
    labeling_rank = np.argmax(labeling_rates, axis=0)

    labeling_winner = np.zeros((spiking_neurons, validation_window))
    for j in range(validation_window):
        labeling_winner[labeling_rank[j], j] = labeling_rates[labeling_rank[j], j]

    number_classes = 10
    neuron_class_rate = np.zeros((spiking_neurons, number_classes))
    for j in range(number_classes):
        class_rate = labeling_winner[:, (labeling_y == j).reshape(validation_window)]
        neuron_class_rate[:, j] = class_rate.sum(axis=1)

    neuron_labels = np.argmax(neuron_class_rate, axis=1)
    np.save(result_path + "partial_label_%06d" % 0, neuron_labels)

    # Test
    validation_rates = np.dot(weights, testing_x.transpose()) / threshold.reshape((spiking_neurons, 1))
    validation_rank = np.argmax(validation_rates, axis=0)

    validation_labels = neuron_labels[validation_rank]

    validation_labels_difference = testing_y.reshape(-1) - validation_labels
    validation_labels_correct = len(np.where(validation_labels_difference == 0)[0])
    validation_accuracy = validation_labels_correct / float(validation_x.shape[0])

    testing_accuracies.append([0, validation_accuracy])


for i in range(total_training_samples):

    input_image = training_x[input_indexes[i], :]

    # get spiking frequency
    gain = np.sum(input_image * weights, axis=1)
    frequency = gain/threshold

    # get winner neuron
    winner_neuron = np.argmax(frequency)
    winner_neuron_frequency = frequency[winner_neuron]
    training_rates[i, winner_neuron] = winner_neuron_frequency

    # get number of spikes
    number_spikes = alpha_number_spikes * running_time * winner_neuron_frequency

    # update weights
    weights[winner_neuron, :] += alpha_weight_update * number_spikes * input_image
    weights[winner_neuron, :] *= weight_total_connection / np.sum(weights[winner_neuron, :])

    # update threshold
    threshold = threshold + (threshold_offset - threshold)/threshold_tau
    threshold[winner_neuron] += threshold_update * number_spikes

    # print i, number_spikes, threshold[winner_neuron]

    if (i+1) % training_print_step == 0:
        loop_time = time.time() - loop_time_start
        print 'runs done:', i+1, 'of', int(total_training_samples), ', required time:', loop_time
        loop_time_start = time.time()

    if (i + 1) in [0, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        np.save(result_path + "partial_weight_%06d" % (i + 1), weights)
        np.save(result_path + "partial_threshold_%06d" % (i + 1), threshold)

    if (((i+1) < 20000) and (((i+1) % 1000) == 0)) or (((i+1) >= 20000) and (((i+1) % 10000) == 0)):
        validation_time_start = time.time()

        # Labeling
        labeling_x = training_x[:validation_window]
        labeling_y = training_y[:validation_window]

        labeling_rates = np.dot(weights, labeling_x.transpose())/threshold.reshape((spiking_neurons, 1))
        labeling_rank = np.argmax(labeling_rates, axis=0)

        labeling_winner = np.zeros((spiking_neurons, validation_window))
        for j in range(validation_window):
            labeling_winner[labeling_rank[j], j] = labeling_rates[labeling_rank[j], j]

        neuron_class_rate = np.zeros((spiking_neurons, number_classes))
        for j in range(number_classes):
            class_rate = labeling_winner[:, (labeling_y == j).reshape(validation_window)]
            neuron_class_rate[:, j] = class_rate.sum(axis=1)

        neuron_labels = np.argmax(neuron_class_rate, axis=1)

        # Validation
        validation_rates = np.dot(weights, validation_x.transpose())/threshold.reshape((spiking_neurons, 1))
        validation_rank = np.argmax(validation_rates, axis=0)

        validation_labels = neuron_labels[validation_rank]

        validation_labels_difference = validation_y.reshape(-1) - validation_labels
        validation_labels_correct = len(np.where(validation_labels_difference == 0)[0])
        validation_accuracy = validation_labels_correct / float(validation_x.shape[0])

        validation_accuracies.append([i+1, validation_accuracy])

        validation_time = time.time() - validation_time_start
        loop_time_start = time.time()

        print "%06d -> %0.4f : gain(%0.4f), spikes(%7.4f), w_update(%0.5f), thr(%8.4f), %0.4f" % \
              (i+1, validation_accuracy, gain[winner_neuron], number_spikes, alpha_weight_update * number_spikes,
               threshold.mean(), validation_time)

    if (i + 1) in [0, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        loop_time = time.time() - loop_time_start
        validation_time_start = time.time()

        # Labeling
        labeling_x = training_x
        labeling_y = training_y

        labeling_rates = np.dot(weights, labeling_x.transpose())/threshold.reshape((spiking_neurons, 1))
        labeling_rank = np.argmax(labeling_rates, axis=0)

        labeling_winner = np.zeros((spiking_neurons, validation_window))
        for j in range(validation_window):
            labeling_winner[labeling_rank[j], j] = labeling_rates[labeling_rank[j], j]

        number_classes = 10
        neuron_class_rate = np.zeros((spiking_neurons, number_classes))
        for j in range(number_classes):
            class_rate = labeling_winner[:, (labeling_y == j).reshape(validation_window)]
            neuron_class_rate[:, j] = class_rate.sum(axis=1)

        neuron_labels = np.argmax(neuron_class_rate, axis=1)
        np.save(result_path + "partial_label_%06d" % (i + 1), neuron_labels)

        # Test
        validation_rates = np.dot(weights, testing_x.transpose())/threshold.reshape((spiking_neurons, 1))
        validation_rank = np.argmax(validation_rates, axis=0)

        validation_labels = neuron_labels[validation_rank]

        validation_labels_difference = testing_y.reshape(-1) - validation_labels
        validation_labels_correct = len(np.where(validation_labels_difference == 0)[0])
        validation_accuracy = validation_labels_correct / float(validation_x.shape[0])

        testing_accuracies.append([i+1, validation_accuracy])

        validation_time = time.time() - validation_time_start
        loop_time_start = time.time()

        print "%06d -> %0.4f : gain(%0.4f), spikes(%3.4f), w_update(%0.5f), thr(%0.4f), %0.4f, %0.4f" % \
              (i+1, validation_accuracy, gain[winner_neuron], number_spikes, alpha_weight_update * number_spikes,
               threshold.mean(), loop_time, validation_time)

training_time = time.time() - training_time_start

np.save(result_path + 'training_indexes', input_indexes)
np.save(result_path + 'training_rates', training_rates)
np.save(result_path + 'training_validation', validation_accuracies)
np.save(result_path + 'training_testing', testing_accuracies)

np.save(result_path + 'training_weight', weights)
np.save(result_path + 'training_threshold', threshold)
np.save(result_path + 'training_time', training_time)


# Labeling
training_x = training_set_x
training_y = training_set_y

number_training_samples = training_x.shape[0]

labeling_time_start = time.time()

labeling_rates = np.dot(weights, training_x.transpose())/threshold.reshape((spiking_neurons, 1))
labeling_rank = np.argsort(labeling_rates, axis=0)[-1, :]

labeling_winner = np.zeros((spiking_neurons, number_training_samples))
for i in range(number_training_samples):
    labeling_winner[labeling_rank[i], i] = 1

neuron_class_rate = np.zeros((spiking_neurons, number_classes))
for i in range(number_classes):
    class_rate = labeling_winner[:, (training_y == i).reshape(number_training_samples)]
    neuron_class_rate[:, i] = class_rate.sum(axis=1)

neuron_labels = np.argsort(neuron_class_rate, axis=1)[:, -1]

print neuron_labels

labeling_time = time.time() - labeling_time_start

np.save(result_path + 'labeling_rates', labeling_rates)
np.save(result_path + 'labeling_labels', neuron_labels)
np.save(result_path + 'labeling_time', labeling_time)


# Testing
number_testing_samples = testing_y.shape[0]

testing_time_start = time.time()

testing_rates = np.dot(weights, testing_x.transpose())/threshold.reshape((spiking_neurons, 1))
testing_rank = np.argsort(testing_rates, axis=0)[-1, :]

testing_labels = neuron_labels[testing_rank]

testing_labels_difference = testing_y.reshape(-1) - testing_labels
testing_labels_correct = len(np.where(testing_labels_difference == 0)[0])
testing_accuracy = testing_labels_correct/float(number_testing_samples)

print 'Accuracy (%s presentations):' % total_training_samples, testing_accuracy

testing_time = time.time() - testing_time_start

print 'training time:', training_time
print 'labeling time:', labeling_time
print 'testing time:', testing_time
print 'total time:', training_time+labeling_time+testing_time

np.save(result_path + 'testing_rates', testing_rates)
np.save(result_path + 'testing_labels', testing_labels)
np.save(result_path + 'testing_accuracy', testing_accuracy)
np.save(result_path + 'testing_time', testing_time)
