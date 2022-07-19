"""Implement a quantum Monte Carlo restricted transverse Ising model """
"""with a longitudinal field quantum Boltzmann machine."""

"""Author: Eric Anschuetz"""

# import external libraries
from sqlite3 import adapt
import numpy as np
import tensorflow as tf



"""Implement a Boltzmann machine."""

# import external libraries
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.python.client import device_lib

class BoltzmannMachine(ABC):
    """A Boltzmann machine."""
    def random_samples(self, num_samples=1024):
        """Draw random samples from the Boltzmann machine."""
        pass

    @staticmethod
    @tf.function
    def mini_batch_generator(train_data, num_epochs, batch_size=16):
        """Create an iterator to generate mini-batches."""
        # batch and shuffle the data
        dataset = tf.data.Dataset.from_tensor_slices(train_data)
        dataset = dataset.shuffle(len(train_data))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(count=num_epochs)

        # determine if there is a GPU
        local_device_protos = device_lib.list_local_devices()
        if any([True for x in local_device_protos if x.device_type == 'GPU']):
            dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))

        # create a data iterator
        iterator = iter(dataset)

        # return an operation to generate mini-batches
        return iterator.get_next()

    @abstractmethod
    def train_op(self, minibatch, optimizer):
        """Create an operation to train the Boltzmann machine."""
        pass

class QMC_TIM_QBM(BoltzmannMachine):
    """A quantum Monte Carlo implementation of a restricted transverse Ising model """
    """with a longitudinal field quantum Boltzmann machine."""
    def __init__(self, visible_nodes, hidden_nodes, initial_params=None,
                 num_replicas=512, num_its=10, betas=None, coreset=False, adaptive=False):
        self.coreset = coreset
        self.adaptive = adaptive
        # set the nodes
        self.visible_nodes = visible_nodes
        self.hidden_nodes = hidden_nodes
        self.num_total_nodes = len(self.visible_nodes) + len(self.hidden_nodes)

        # set the Hamiltonian parameters
        if initial_params is None:
            initial_params = {}
        if 'gammas' not in initial_params:
            initial_params['gammas'] = np.random.normal(loc=1,
                                                        scale=0.005,
                                                        size=self.num_total_nodes).astype(
                                                            np.float32)
        if 'biases' not in initial_params:
            initial_params['biases'] = np.random.normal(scale=0.005,
                                                        size=self.num_total_nodes).astype(
                                                            np.float32)
        if 'weights' not in initial_params:
            initial_params['weights'] = np.random.normal(scale=0.01,
                                                         size=(len(self.visible_nodes),
                                                               len(self.hidden_nodes))).astype(
                                                                   np.float32)
        self.params = {}
        self.params['gammas'] = tf.Variable(initial_value=initial_params['gammas'], name='gammas')
        self.params['biases'] = tf.Variable(initial_value=initial_params['biases'], name='biases')
        self.params['weights'] = tf.Variable(initial_value=initial_params['weights'], name='weights')
        self.flattened_weights = tf.concat([tf.map_fn(lambda vw: tf.concat([tf.zeros(
            len(self.visible_nodes)), vw], 0),
                                                      self.params['weights']),
                                            tf.map_fn(lambda hw: tf.concat(
                                                [hw, tf.zeros(len(self.hidden_nodes))], 0),
                                                      tf.transpose(self.params['weights']))], 0)

        # set the QMC parameters
        self.num_replicas = num_replicas
        self.num_its = num_its
        if betas is None:
            self.betas = tf.constant(np.array([i/5 for i in range(6)]).astype(np.float32))
        else:
            self.betas = tf.constant(betas)
        self.q_adj_mat = tf.constant(np.array([[[1. if abs(i-j) == 1
                                                 or abs(i-j) == self.num_its-1 else 0.
                                                 for j in range(self.num_its)]
                                                for i in range(self.num_its)]
                                               for _ in range(self.num_total_nodes)])
                                     .astype(np.float32))

        # replicate the system and look at the first imaginary time slice
        self.replicas = self.pa()[:, 0]

    @tf.function
    def random_samples(self, num_samples=1024):
        """Draw random samples from the QMC TIM QBM."""
        return self.replicas[:num_samples]

    @tf.function
    def beta_energy(self, z, beta, epsilon=1e-5):
        """Find beta times the energy."""
        # find the classical energy
        classical_energy_slices = tf.add(tf.einsum('i,ji->j', self.params['biases'], z),
                                         tf.einsum('ij,ki,kj->k', self.params['weights'],
                                                   tf.gather(z, list(self.visible_nodes), axis=1),
                                                   tf.gather(z, list(self.hidden_nodes), axis=1)))
        classical_energy = tf.multiply(beta, tf.reduce_mean(classical_energy_slices),
                                       name='classical_energy')

        # find the quantum energy
        scaled_gammas = tf.tanh(tf.divide(tf.multiply(beta, self.params['gammas']), self.num_its))
        log_args = tf.maximum(epsilon, scaled_gammas)
        quantum_energy_weights = tf.multiply(0.5, tf.math.log(log_args))
        bulk_quantum_energy = tf.einsum('ij,ij->',
                                        tf.multiply(quantum_energy_weights, z[:-1]), z[1:])
        quantum_energy = tf.add(bulk_quantum_energy, tf.einsum('i,i->', tf.multiply(
            quantum_energy_weights, z[-1]), z[0]),
                                name='quantum_energy')

        # return the total energy
        return tf.add(classical_energy, quantum_energy, name='energy')

    @tf.function
    def pimc(self, i, replicas, beta, beta_qe_matrix, rands, epsilon=0.1):
        """Perform path integral Monte Carlo."""
        # calculate the bias energy changes
        ind = tf.cast(i, tf.int32)
        bias_change = tf.divide(tf.multiply(-2., tf.multiply(
            beta, self.params['biases'])), self.num_its)

        def its_update(i):
            """Update an imaginary time slice's spins."""
            # set the imaginary time slice index
            its_ind = tf.cast(i, tf.int32)

            # calculate the weight energy changes
            weight_change = tf.divide(tf.multiply(-4., tf.tensordot(
                replicas[ind, its_ind],
                tf.multiply(beta, self.flattened_weights), 1)), self.num_its)

            def local_update(j):
                """Update a spin."""
                # set the site index
                site_ind = tf.cast(j, tf.int32)

                # set the change in beta * energy
                classical_beta_energy_change = tf.multiply(tf.add(
                    bias_change[site_ind], weight_change[site_ind]),
                                                           replicas[ind, its_ind, site_ind])
                quantum_beta_energy_change = tf.multiply(tf.multiply(
                    -2., tf.tensordot(beta_qe_matrix[site_ind, its_ind],
                                      replicas[ind, :, site_ind], 1)),
                                                         replicas[ind, its_ind, site_ind])
                beta_energy_change = tf.add(classical_beta_energy_change,
                                            quantum_beta_energy_change)

                # determine whether or not to update the site
                update = tf.cast(tf.logical_or(tf.greater(
                    tf.negative(beta_energy_change), 1/epsilon),
                                               tf.less(rands[ind, its_ind, site_ind],
                                                       tf.exp(tf.negative(beta_energy_change)))),
                                 tf.float32)
                with tf.control_dependencies([replicas[ind, its_ind, site_ind].assign(
                        tf.multiply(tf.subtract(1., tf.multiply(2., update)),
                                    replicas[ind, its_ind, site_ind]))]):
                    return replicas[ind, its_ind, site_ind]

            # evaluate over every spin in the imaginary time slice
            return tf.map_fn(local_update, tf.range(
                tf.cast(self.num_total_nodes, tf.float32)), parallel_iterations=1)

        # evaluate over every imaginary time slice
        return tf.map_fn(its_update, tf.range(
            tf.cast(self.num_its, tf.float32)), parallel_iterations=1)

    def pa(self):
        """Perform population annealing."""
        # define replicas of the system
        replicas_init = tf.tile(tf.expand_dims(tf.subtract(1., tf.multiply(2., tf.cast(
            tf.random.uniform([2*self.num_replicas,
                               self.num_total_nodes],
                              maxval=2, dtype=tf.int32), tf.float32))), axis=1),
                                [1, self.num_its, 1])
        with tf.compat.v1.variable_scope('replica_scope', reuse=tf.compat.v1.AUTO_REUSE):
            replicas = tf.Variable(initial_value=replicas_init, name='replicas')

        @tf.function
        def anneal(i, current_num_replicas, epsilon=1e-5, exp_arg_max=10.):
            """Anneal the given replicas beginning at the given inverse temperature index."""
            # find the Boltzmann weights of the replicas
            bws = tf.maximum(epsilon,
                             tf.map_fn(lambda z: tf.exp(tf.minimum(
                                 exp_arg_max,
                                 tf.subtract(self.beta_energy(z, self.betas[i]),
                                             self.beta_energy(z, self.betas[tf.add(i, 1)])))),
                                       replicas[:current_num_replicas]), name='bws')
            total_bw = tf.reduce_sum(bws, name='total_bw')

            # calculate the partition function ratio
            pfr = tf.divide(total_bw, tf.cast(self.num_replicas, tf.float32), name='pfr')

            # find the number of replicas to make
            mean_num_replicas = tf.divide(bws, pfr, name='rws')
            def get_new_num_replica(j):
                """Get the new number of replicas for replica i."""
                ind = tf.cast(j, tf.int32)
                return tf.cond(tf.less(tf.random.uniform([]),
                                       tf.subtract(mean_num_replicas[ind],
                                                   tf.floor(mean_num_replicas[ind]))),
                               true_fn=lambda: tf.math.ceil(mean_num_replicas[ind]),
                               false_fn=lambda: tf.math.floor(mean_num_replicas[ind]))
            new_num_replicas = tf.cast(tf.map_fn(get_new_num_replica,
                                                 tf.range(tf.cast(tf.reduce_prod(
                                                     tf.shape(mean_num_replicas)),
                                                                  tf.float32))), tf.int32)

            # replicate the replicas
            def replicate(j, new_replicas):
                """Replicate the replicas."""
                return tf.add(j, 1), tf.concat([new_replicas,
                                                tf.tile(tf.expand_dims(replicas[j], axis=0),
                                                        [new_num_replicas[j], 1, 1])], 0)
            initial_replication = tf.tile(tf.expand_dims(replicas[0], axis=0),
                                          [new_num_replicas[0], 1, 1])
            _, expanded_replicas = tf.while_loop(
                lambda j, _: tf.less(j, current_num_replicas),
                replicate,
                [1, initial_replication],
                shape_invariants=[tf.TensorShape([]),
                                  tf.TensorShape([None,
                                                  self.num_its,
                                                  self.num_total_nodes])])
            current_num_replicas = tf.reduce_sum(new_num_replicas)
            expanded_replicas = tf.reshape(expanded_replicas, [current_num_replicas,
                                                               self.num_its, self.num_total_nodes])

            # set the imaginary time weights
            log_args = tf.maximum(epsilon, tf.tanh(
                tf.divide(tf.multiply(self.betas[tf.add(i, 1)],
                                      self.params['gammas']), self.num_its)))
            beta_qe_matrix = tf.einsum('i,ijk->ijk',
                                       tf.multiply(0.5, tf.math.log(log_args)), self.q_adj_mat)

            # perform one PIMC step
            rands = tf.random.uniform([current_num_replicas, self.num_its, self.num_total_nodes])
            with tf.control_dependencies([
                    replicas[:current_num_replicas].assign(expanded_replicas)]):
                equil_replicas = tf.map_fn(lambda j: self.pimc(
                    j, replicas, self.betas[tf.add(i, 1)],
                    beta_qe_matrix, rands),
                                           tf.range(tf.cast(current_num_replicas, tf.float32)))
            with tf.control_dependencies([
                    replicas[:current_num_replicas].assign(equil_replicas)]):
                return tf.add(i, 1), current_num_replicas

        # anneal
        with tf.control_dependencies([replicas.assign(replicas_init)]):
            _, current_num_replicas = tf.while_loop(lambda i, _:
                                                    tf.less(i, np.prod(
                                                        self.betas.get_shape().as_list())-1),
                                                    anneal,
                                                    [0, self.num_replicas])
            with tf.control_dependencies([current_num_replicas]):
                replicas = tf.reshape(replicas,
                                      [-1,
                                       self.num_its,
                                       self.num_total_nodes],
                                      name='replicas')

                # return the annealed replicas
                return replicas[:current_num_replicas]

    @tf.function
    def positive_phase(self, data):
        """Compute the positive phase of the gradient."""

        if self.coreset:
            batch = data
            pts = batch[:,1:]
            weights = batch[:,0]
            total_weight = tf.reduce_sum(weights)

            # compute the effective biases
            effective_biases = tf.add(tf.gather(self.params['biases'], list(self.hidden_nodes)),
                                    tf.einsum('ij,jk->ik', pts, self.params['weights']))

            # compute the effective quantum scaling
            effective_qs = tf.sqrt(tf.add(tf.square(tf.gather(self.params['gammas'],
                                                            list(self.hidden_nodes))),
                                        tf.square(effective_biases)))

            # compute the clamped correlation functions
            exp_hidden_biases = tf.negative(tf.multiply(tf.divide(effective_biases,
                                                                effective_qs),
                                                        tf.tanh(effective_qs)))
            exp_weights = tf.einsum('ij,ik->ijk', pts, exp_hidden_biases)

            avg_pts = tf.divide(tf.einsum('i,ij->j', weights, pts), total_weight)
            avg_exp_hidden_biases = tf.divide(tf.einsum('i,ij->j', weights, exp_hidden_biases), total_weight)
            avg_exp_weights = tf.divide(tf.einsum('i,ijk->jk', weights, exp_weights), total_weight)

            # return the positive phases
            return (tf.concat([avg_pts, tf.multiply(-1.0, avg_exp_hidden_biases)], 0), avg_exp_weights)

        else:
            # compute the effective biases
            effective_biases = tf.add(tf.gather(self.params['biases'], list(self.hidden_nodes)),
                                    tf.einsum('ij,jk->ik', data, self.params['weights']))

            # compute the effective quantum scaling
            effective_qs = tf.sqrt(tf.add(tf.square(tf.gather(self.params['gammas'],
                                                            list(self.hidden_nodes))),
                                        tf.square(effective_biases)))

            # compute the clamped correlation functions
            exp_hidden_biases = tf.negative(tf.multiply(tf.divide(effective_biases,
                                                                effective_qs),
                                                        tf.tanh(effective_qs)))
            exp_weights = tf.einsum('ij,ik->ijk', data, exp_hidden_biases)

            # return the positive phases
            return (tf.concat([tf.reduce_mean(data, axis=0),
                            tf.multiply(-1.0, tf.reduce_mean(exp_hidden_biases, axis=0))], 0),
                    tf.reduce_mean(exp_weights, axis=0))

    @tf.function
    def negative_phase(self):
        """Compute the negative phase of the gradient."""
        # calculate the expectation values of various observables
        exp_biases = tf.multiply(-1.0, tf.reduce_mean(self.replicas, axis=0))
        exp_weights = tf.reduce_mean(tf.einsum('ij,ik->ijk',
                                               self.replicas[:, :len(self.visible_nodes)],
                                               self.replicas[:, len(self.visible_nodes):]), axis=0)
        

        # return the negative phases
        return exp_biases, exp_weights

    def gen_grads_and_vars(self, data):
        """Generate the grads_and_vars list for the QMC TIM QBM."""
        # calculate the positive and negative phases
        bias_pps, weight_pps = self.positive_phase(data)
        bias_nps, weight_nps = self.negative_phase()

        # calculate and return the gradients
        bias_grads = tf.subtract(bias_pps, bias_nps)
        weight_grads = tf.subtract(weight_pps, weight_nps)
        self.bias_grads = bias_grads
        self.weight_grads = weight_grads
        return [(bias_grads, self.params['biases']), (weight_grads, self.params['weights'])]

    def train_op(self, minibatch, optimizer):
        """Create an operation to train the QMC TIM QBM."""
        return optimizer.apply_gradients(self.gen_grads_and_vars(minibatch))

