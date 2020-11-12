'''
Labeling, simplified Competitive Spiking Neural Network with 400 integrate and fire spiking neurons, trace based STDP,
input weight normalization, direct inhibition, no resting time
Built on Peter Diehl's implementation https://github.com/peter-u-diehl/stdp-mnist.git

@author: Anonymous et al.
'''



import time
import sys
import numpy as np
import xmltodict

import brian_no_units
import brian as b
from brian import ms


# IMPORT PARAMETERS
parameters = "../../params/CSNN-MNIST-100.xml"
result_path = "../../results/CSNN-MNIST-100.xml"
partial_number = 0
if len(sys.argv) == 4:
    parameters = str(sys.argv[1])
    result_path = str(sys.argv[2])
    partial_number = int(sys.argv[3])

with open(parameters) as fd:
    params = xmltodict.parse(fd.read())


spiking_neurons = int(params['params']['network']['spiking_neurons'])
running_time = int(params['params']['network']['running_time'])

threshold_0 = float(params['params']['network']['threshold_0'])
threshold_offset = float(params['params']['network']['threshold_offset'])
threshold_update = float(params['params']['network']['threshold_update'])
threshold_tau = float(params['params']['network']['threshold_tau'])

validation_print_step = int(params['params']['validation']['print_step'])
validation_number_samples = int(params['params']['validation']['validation_window'])

training_data_x_path = str(params['params']['dataset']['training_x']['@path'])
training_data_y_path = str(params['params']['dataset']['training_y']['@path'])

print 'EXPERIMENT', result_path


# BRIAN SET UP
b.set_global_preferences(
    defaultclock=b.Clock(dt=0.1 * b.ms),  # The default clock to use if none is provided.
    useweave=True,  # Defines whether or not functions should use inlined compiled C code where defined.
    gcc_options=['-march=native'],  # Defines the compiler switches passed to the gcc compiler.
    usecodegen=True,  # Whether or not to use experimental code generation support.
    usecodegenweave=True,  # Whether or not to use C with experimental code generation support.
    usecodegenstateupdate=True,  # Whether or not to use experimental code generation support on state updaters.
    usecodegenthreshold=False,  # Whether or not to use experimental code generation support on thresholds.
    usenewpropagate=True,  # Whether or not to use experimental new C propagation functions.
    usecstdp=True,  # Whether or not to use experimental new C STDP.
)


# LOAD DATA
load_threshold = np.load(result_path + '/partial_threshold_%06d.npy'%partial_number)
load_weights = np.load(result_path + '/partial_weight_%06d.npy'%partial_number)
load_delay = np.load(result_path + '/training_delay.npy')


# LOAD MNIST TRAINING SET
training_set_x = np.load(training_data_x_path)
training_set_x = training_set_x.astype('float')/8.0
training_set_x = training_set_x.reshape(-1, training_set_x.shape[1]*training_set_x.shape[2])

training_set_y = np.load(training_data_y_path)

training_x = training_set_x[:validation_number_samples, :]
training_y = training_set_y[:validation_number_samples]


# DEFINE SENSORY GROUP
sensory_neurons = training_x.shape[1]
neuron_groups = {'poisson': b.PoissonGroup(sensory_neurons, 0)}


# DEFINE SPIKING GROUP

# model equations
v_rest_e = -65. * b.mV
v_reset_e = -65. * b.mV
v_threshold_e = -52. * b.mV

tau_v = 100 * b.ms
tau_ge = 1.0 * b.ms
tau_gi = 2.0 * b.ms
threshold_tau = threshold_tau * b.ms

time_refractory_e = 5. * b.ms

neuron_eqs_e = '''
        dv/dt = ((v_rest_e-v) + ge*-v + gi*(-100.*mV-v)) / (tau_v)  : volt
        dge/dt = -ge/tau_ge                                         : 1
        dgi/dt = -gi/tau_gi                                         : 1
        dthreshold/dt = -threshold/(threshold_tau)                  : volt
        dtimer/dt = 1                                               : ms
        '''

# reset equations
threshold_update = threshold_update * b.mV
reset_eqs_e = 'v = v_reset_e; threshold += threshold_update; timer = 0*ms'

# threshold equations
threshold_0 = threshold_0 * b.mV
threshold_offset = threshold_offset * b.mV
threshold_eqs_e = '(v>(threshold - threshold_offset + v_threshold_e)) * (timer>time_refractory_e)'

# group instantiation
neuron_groups['spiking'] = b.NeuronGroup(N=spiking_neurons, model=neuron_eqs_e, threshold=threshold_eqs_e,
                                         refractory=time_refractory_e, reset=reset_eqs_e, compile=True, freeze=True)

neuron_groups['spiking'].v = v_rest_e
neuron_groups['spiking'].threshold = load_threshold


# DEFINE CONNECTIONS
connections = {}

# sensory -> spiking neurons
weight_matrix_input = load_weights

delay_input_excitatory = (0 * b.ms, 10 * b.ms)

connections['input'] = b.Connection(neuron_groups['poisson'], neuron_groups['spiking'], structure='dense',
                                    state='ge', delay=True, max_delay=delay_input_excitatory[1])
connections['input'].connect(neuron_groups['poisson'], neuron_groups['spiking'], weight_matrix_input,
                             delay=delay_input_excitatory)

connections['input'].delay[:, :] = load_delay

# lateral inhibition
weight_matrix_inhibitory = np.ones(spiking_neurons) - np.identity(spiking_neurons)
weight_matrix_inhibitory *= 17.0

connections['inhibitory'] = b.Connection(neuron_groups['spiking'], neuron_groups['spiking'], structure='dense',
                                         state='gi')
connections['inhibitory'].connect(neuron_groups['spiking'], neuron_groups['spiking'], weight_matrix_inhibitory)


# RUN LABELING
input_intensity = 2.0
default_input_intensity = input_intensity

neuron_groups['poisson'].rate = 0
b.run(0 * b.ms)

spike_counter = b.SpikeCounter(neuron_groups['spiking'])
previous_spike_count = np.zeros(spiking_neurons)
result_spike_activity = np.zeros((validation_number_samples, spiking_neurons))

total_start = time.time()
start = time.time()

i = 0
while i < validation_number_samples:

    # present one image
    neuron_groups['poisson'].rate = training_x[i, :] * input_intensity

    b.run(running_time * b.ms)

    # evaluate neuron activity
    current_spike_count = spike_counter.count - previous_spike_count
    previous_spike_count = np.copy(spike_counter.count)

    number_spikes = np.sum(current_spike_count)
    # print i, number_spikes

    if number_spikes < 5:
        input_intensity += 1
    else:
        if (i + 1) % validation_print_step == 0:
            end = time.time()
            print 'runs done:', i + 1, 'of', int(validation_number_samples), ', required time:', end - start
            start = time.time()

        result_spike_activity[i, :] = current_spike_count

        input_intensity = default_input_intensity
        i += 1

# save results
total_time = time.time() - total_start
print 'Total required time:', total_time
# np.save(result_path + '/labeling_rates', result_spike_activity)
np.save(result_path + '/validation_labeling_time_%06d' % partial_number, total_time)


# GET NEURON LABELS

number_classes = 10
total_class_activity = np.zeros((number_classes, spiking_neurons))

# get neuron activity per class
for i in xrange(number_classes):
    class_spike_activity = result_spike_activity[training_y[:validation_number_samples].reshape(-1) == i, :]
    total_class_activity[i, :] = np.sum(class_spike_activity, axis=0)/float(class_spike_activity.shape[0])

# get max class activity
class_rank = np.argsort(total_class_activity, axis=0)
neuron_labels = class_rank[-1, :]

print partial_number
print neuron_labels

np.save(result_path + '/validation_labeling_labels_%06d' % partial_number, neuron_labels)
