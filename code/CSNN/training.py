'''
Training, simplified Competitive Spiking Neural Network with integrate and fire spiking neurons, trace based STDP,
input weight normalization, direct inhibition, no resting time
Built on Peter Diehl's implementation https://github.com/peter-u-diehl/stdp-mnist.git

@author: Anonymous et al.
'''


import sys
import time
import numpy as np
import xmltodict
import cv2

import brian_no_units
import brian as b
from brian import ms


# IMPORT PARAMETERS
parameters = "../../params/CSNN-MNIST-100.xml"
result_path = "../../results/CSNN-MNIST-100.xml"
if len(sys.argv) == 3:
    parameters = str(sys.argv[1])
    result_path = str(sys.argv[2])

with open(parameters) as fd:
    params = xmltodict.parse(fd.read())

spiking_neurons = int(params['params']['network']['spiking_neurons'])
running_time = int(params['params']['network']['running_time'])

threshold_0 = float(params['params']['network']['threshold_0'])
threshold_offset = float(params['params']['network']['threshold_offset'])
threshold_update = float(params['params']['network']['threshold_update'])
threshold_tau = float(params['params']['network']['threshold_tau'])

weight_blurring_number = int(params['params']['network']['weight_initialization_blurring'])
weight_total_connection = float(params['params']['network']['weight_total_value'])

training_number_epochs = int(params['params']['training']['number_epochs'])
training_print_step = int(params['params']['training']['print_step'])
training_save_step = int(params['params']['training']['save_step'])

validation_number_samples = int(params['params']['validation']['number_samples'])

training_data_x_path = str(params['params']['dataset']['training_x']['@path'])
training_data_y_path = str(params['params']['dataset']['training_y']['@path'])

with open(result_path + 'parameters.xml', 'w') as param_file:
    param_file.write(xmltodict.unparse(params))

print 'EXPERIMENT', result_path


# BRIAN SET UP
use_stdp = True
if spiking_neurons > 600:
    use_stdp = False
b.set_global_preferences(
    defaultclock=b.Clock(dt=0.1 * b.ms),  # The default clock to use if none is provided.
    useweave=True,  # Defines whether or not functions should use inlined compiled C code where defined.
    gcc_options=['-march=native'],  # Defines the compiler switches passed to the gcc compiler.
    usecodegen=True,  # Whether or not to use experimental code generation support.
    usecodegenweave=True,  # Whether or not to use C with experimental code generation support.
    usecodegenstateupdate=True,  # Whether or not to use experimental code generation support on state updaters.
    usecodegenthreshold=False,  # Whether or not to use experimental code generation support on thresholds.
    usenewpropagate=True,  # Whether or not to use experimental new C propagation functions.
    usecstdp=use_stdp,  # Whether or not to use experimental new C STDP.
)


# LOAD TRAINING SET
training_set_x = np.load(training_data_x_path)
image_width = training_set_x.shape[1]
image_height = training_set_x.shape[2]

training_set_x = training_set_x.astype('float')/8.0
training_set_x = training_set_x.reshape(-1, training_set_x.shape[1]*training_set_x.shape[2])

training_set_y = np.load(training_data_y_path)

training_x = training_set_x[:-validation_number_samples, :]
training_y = training_set_y[:-validation_number_samples]


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


# DEFINE SENSORY GROUP
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
neuron_groups['spiking'].threshold = threshold_0

# save initial threshold
np.save(result_path + "/partial_threshold_%06d" % 0, neuron_groups['spiking'].threshold)

# DEFINE CONNECTIONS
connections = {}

# sensory -> spiking neurons (sample based initialization)
initial_weights = np.copy(training_x[-spiking_neurons:])

blurring_number = weight_blurring_number
if blurring_number > 0:
    for i in range(spiking_neurons):
        blur = cv2.blur(initial_weights[i].reshape(image_width, image_height), (blurring_number, blurring_number))
        initial_weights[i] = blur.reshape(1, sensory_neurons)
initial_weights = initial_weights.transpose()

# # sensory -> spiking neurons (random initialization)
# initial_weights = np.random.random((sensory_neurons, spiking_neurons))
# initial_weights += 0.01
# initial_weights *= 0.3

# normalize
total_input_weight_neuron = np.sum(initial_weights, axis=0)
initial_weights *= weight_total_connection / total_input_weight_neuron

# save initial weights
np.save(result_path + "/partial_weight_%06d" % 0, initial_weights)

# create connection
delay_input_excitatory = (0 * b.ms, 10 * b.ms)
connections['input'] = b.Connection(neuron_groups['poisson'], neuron_groups['spiking'], structure='dense',
                                    state='ge', delay=True, max_delay=delay_input_excitatory[1])
connections['input'].connect(neuron_groups['poisson'], neuron_groups['spiking'], initial_weights,
                             delay=delay_input_excitatory)

# save delay
np.save(result_path + '/training_delay', connections['input'].delay)

# lateral inhibition
weight_matrix_inhibitory = np.ones(spiking_neurons) - np.identity(spiking_neurons)
weight_matrix_inhibitory *= 17.0

connections['inhibitory'] = b.Connection(neuron_groups['spiking'], neuron_groups['spiking'], structure='dense',
                                         state='gi')
connections['inhibitory'].connect(neuron_groups['spiking'], neuron_groups['spiking'], weight_matrix_inhibitory)


# DEFINE STDP
stdp_connections = {}

tc_pre_ee = 20 * b.ms
tc_post_1_ee = 20 * b.ms
tc_post_2_ee = 40 * b.ms
nu_ee_pre = 0.0001
nu_ee_post = 0.01

stdp_eqs = '''
                post2before                             : 1.0
                dpre/dt = -pre/(tc_pre_ee)              : 1.0
                dpost1/dt = -post1/(tc_post_1_ee)       : 1.0
                dpost2/dt = -post2/(tc_post_2_ee)       : 1.0
           '''

stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'

wmax_ee = 1.0

stdp_connections['input'] = b.STDP(connections['input'], eqs=stdp_eqs, pre=stdp_pre_ee, post=stdp_post_ee,
                                   wmin=0., wmax=wmax_ee)


# TRAINING PROCESS
input_intensity = 2.0
default_input_intensity = input_intensity

i = 0
neuron_groups['poisson'].rate = 0
b.run(0 * b.ms)

spike_counter = b.SpikeCounter(neuron_groups['spiking'])
previous_spike_count = np.zeros(spiking_neurons)
result_spike_activity = np.zeros((total_training_samples, spiking_neurons))

total_start = time.time()
start = time.time()
while i < total_training_samples:

    # present sample
    neuron_groups['poisson'].rate = training_x[input_indexes[i], :] * input_intensity

    b.run(running_time * b.ms)

    # normalize connection weights
    total_input_weight_neuron = np.sum(connections['input'].W, axis=0)
    connections['input'].W *= weight_total_connection / total_input_weight_neuron

    # evaluate neuron activity
    current_spike_count = spike_counter.count - previous_spike_count
    previous_spike_count = np.copy(spike_counter.count)

    number_spikes = np.sum(current_spike_count)
    # print i, number_spikes

    if number_spikes < 5:
        input_intensity += 1
        print input_intensity
    else:
        result_spike_activity[i, :] = current_spike_count

        # print training time
        if (i+1) % training_print_step == 0:
            end = time.time()
            print 'runs done:', i+1, 'of', int(total_training_samples), ', required time:', end - start, ', spikes: ', number_spikes
            start = time.time()

        # save weights
        if (i + 1) < 20000:
            if (i + 1) % 1000 == 0:
                np.save(result_path + "/partial_weight_%06d" % (i + 1), connections['input'].W)
                np.save(result_path + "/partial_threshold_%06d" % (i + 1), neuron_groups['spiking'].threshold)
        else:
            if (i + 1) % training_save_step == 0:
                np.save(result_path + "/partial_weight_%06d" % (i + 1), connections['input'].W)
                np.save(result_path + "/partial_threshold_%06d" % (i + 1), neuron_groups['spiking'].threshold)


        # jump to next sample
        input_intensity = default_input_intensity
        i += 1

total_time = time.time() - total_start
print 'Total required time:', total_time

# save results
np.save(result_path + '/training_weights', connections['input'].W)
np.save(result_path + '/training_threshold', neuron_groups['spiking'].threshold)

np.save(result_path + '/training_rates', result_spike_activity)
np.save(result_path + '/training_indexes', input_indexes)
np.save(result_path + '/training_time', total_time)
