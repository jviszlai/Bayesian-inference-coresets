import numpy as np
import tensorflow as tf
import scipy
import qiskit
from qiskit.opflow.primitive_ops import PauliSumOp

from active_coresets.qmc_tim_qbm import QMC_TIM_QBM

class PAMC_QBM():

    def __init__(self, visible: np.ndarray, hidden: np.ndarray, gamma: float, population_size: int):
        self.visible = visible
        self.hidden = hidden
        self.biases = np.zeros(len(visible) + len(hidden)).astype(np.float32)
        self.weights = np.random.normal(scale=0.01, size=(len(visible), len(hidden))).astype(np.float32)
        self.gamma = gamma
        self.population_size = population_size
        self.population = -2 * np.random.randint(0, 2, size=(population_size, len(visible) + len(hidden))) + np.ones((population_size, len(visible) + len(hidden)))

    def construct_H(self):
        num_qubits = len(self.visible) + len(self.hidden)

        pauli_list = []

        for node, param in enumerate(self.biases):
            pauli = ['I'] * num_qubits
            pauli[node] = 'Z'
            pauli_list.append((''.join(pauli), -1*param))
            #pauli_list.append((''.join(reversed(pauli)), -1*param))
        
        # double terms
        for a in range(len(self.visible)):
            for b in range(len(self.hidden)):
                param = self.weights[a,b]
                pauli = ['I'] * num_qubits
                pauli[a] = 'Z'
                pauli[b + len(self.visible)] = 'Z'
                pauli_list.append((''.join(pauli), -1*param))
                #pauli_list.append((''.join(reversed(pauli)), -1*param))

        for node in range(num_qubits):
            pauli = ['I'] * num_qubits
            pauli[node] = 'X'
            pauli_list.append((''.join(pauli), -1*self.gamma))
            #pauli_list.append((''.join(reversed(pauli)), -1*self.gamma))
        
        return PauliSumOp.from_list(pauli_list)
    
    def b_eff(self, i, data):
        b_eff = self.biases[i]
        for a in range(len(self.visible)):
            b = i - len(self.visible)
            b_eff += self.weights[a, b] * data[a]
        return b_eff

    def calculate_exps(self, beta, batch):
        num_qubits = len(self.visible) + len(self.hidden)

        H = self.construct_H()

        H = H.mul(-1 * beta)
        exp_H = scipy.linalg.expm(H.to_matrix())
        partition_function = np.trace(exp_H)
        exact_thermal_state = qiskit.quantum_info.DensityMatrix(exp_H / partition_function)

        z = np.zeros(num_qubits).astype(np.float32)
        zz = np.zeros((len(self.visible), len(self.hidden))).astype(np.float32)

        z_pos = np.zeros(num_qubits).astype(np.float32)
        zz_pos = np.zeros((len(self.visible), len(self.hidden))).astype(np.float32)

        for i in range(num_qubits):
            pauli = ['I'] * num_qubits
            pauli[i] = 'Z'
            z[i] = exact_thermal_state.expectation_value(PauliSumOp.from_list([(''.join(pauli), 1)])).real
            #z[i] = exact_thermal_state.expectation_value(PauliSumOp.from_list([(''.join(reversed(pauli)), 1)])).real

            for data in batch:
                if i < len(self.visible):
                    z_pos[i] += data[i]
                else:
                    b_eff = self.b_eff(i, data)
                    D = np.sqrt(self.gamma ** 2 + b_eff ** 2)
                    z_pos[i] += b_eff * np.tanh(D) / D 
            z_pos[i] /= batch.shape[0]

        for i in range(len(self.visible)):
            for j in range(len(self.hidden)):
                pauli = ['I'] * num_qubits
                pauli[i] = 'Z'
                pauli[j + len(self.visible)] = 'Z'
                zz[i,j] = exact_thermal_state.expectation_value(PauliSumOp.from_list([(''.join(pauli), 1)])).real
                #zz[i,j] = exact_thermal_state.expectation_value(PauliSumOp.from_list([(''.join(reversed(pauli)), 1)])).real

                zz_pos[i,j] = z_pos[i] * z_pos[j + len(self.visible)]
        
        return z, zz, z_pos, zz_pos
    
    def exact_dist(self, H, beta):

        def gen_states(n):
            bitstrings = []
            for i in range(int(2**n)):
                bitstrings.append(tuple(f'{i:0{n}b}'))
            return bitstrings

        H = H.mul(-1 * beta)
        exp_H = scipy.linalg.expm(H.to_matrix())
        partition_function = np.trace(exp_H)
        rho = qiskit.quantum_info.DensityMatrix(exp_H / partition_function)

        num_units = len(self.visible) + len(self.hidden)
        
        probability_dist = {}
        for i, visible_state in enumerate(gen_states(len(self.visible))):
            # Construct the projector matrix that will project rho onto the
            # states with the correct visible state
            projector = []
            for j, total_state in enumerate(gen_states(num_units)):
                row = np.zeros(int(2**num_units)) # Constructing an exponentially large matrix...
                cur_vis_state = list(total_state)[:len(self.visible)] # index into the tuple to get the correct ordering
                cur_vis_state = tuple(cur_vis_state) 
                if cur_vis_state == visible_state:
                    row[j] = 1
                projector.append(row)

            projector = qiskit.quantum_info.Operator(np.array(projector))
            probability_dist[i] = np.trace(projector.dot(rho))
        
        return probability_dist
    
    def log_likelihood(self, data, model_dist):
        """
        A well trained QBM minimizes the KL-divergence between the data and model probability distributions.
        
            KL(P_data || P_model) = Sum_v( P_v_data * log(P_v_data / P_v_model) )
            
        Since P_v_data is a constant, minimizing the KL divergence is equivalent to maximizing the average log-likelihood.
        Or, minimizing the average negative log-likelihood (Eq. 3 and 17 of Amin et al.)
        
            average_negative_log_likelihood = - Sum_v( P_v_data * log(P_v_model))
        """
        def data_to_dict(data):
            dict = {}
            for z_pt in data:
                bitstring = ''.join(['0' if b == 1 else '1' for b in z_pt])
                pt = int(bitstring, 2)
                if pt in dict:
                    dict[pt] += 1
                else:
                    dict[pt] = 1
            return {k: v / sum(dict.values()) for k, v in dict.items()}

        num_states = 2**(len(self.visible) + len(self.hidden))
        data_dist = data_to_dict(data)
        log_likelihood = 0.0
        for visible_state in data_dist.keys():
            p_v_data = data_dist.get(visible_state, 0)
            p_v_model = model_dist.get(visible_state, 0)
            if p_v_model > 0:
                log_likelihood -= p_v_data * np.log(p_v_model)
        
        return log_likelihood


    def train(self, data: np.ndarray, epochs: int, batch_size, betas: np.ndarray, num_its: int = 10, learning_rate: float = 0.001, exact=False, v_interval=20, debug=False):
        pv_biases = -0.5 * (data.mean(axis=0) - 1)
        init_v_biases = np.log(pv_biases / (np.ones(pv_biases.shape[0]) - pv_biases))
        self.biases[:len(self.visible)] = init_v_biases

        pop_history = [self.population]

        gammas = np.ones(len(self.visible) + len(self.hidden)) * self.gamma

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.9)

        '''
        quantum_lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(2e-3, 500, power=0.5)
        tf.optimizers.Adam(learning_rate=quantum_lr_schedule, beta_1=0.5, beta_2=0.9)
        '''

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
            for i in range(0, data.shape[0], batch_size):
                H = self.construct_H()
                model_dist = self.exact_dist(H, betas[-1])

                if (i // batch_size + 1) % v_interval == 0:
                    print(f'...Batch {i // batch_size + 1} / {data.shape[0] // batch_size}')
                    print(f'...KL: {self.log_likelihood(data, model_dist)}')
                    if exact:
                        pop_history.append(model_dist)
                    else:
                        pop_history.append(self.population)

                batch = data[i:i + batch_size]
                if exact:
                    '''
                    Exact training 
                    '''
                    

                    bias_nps, weight_nps, bias_pps, weight_pps = self.calculate_exps(betas[-1], batch)

                    self.biases += learning_rate * (bias_pps - bias_nps)
                    self.weights += learning_rate * (weight_pps - weight_nps)

                else:
                    '''
                    Approx training with population annealed path integral monte carlo
                    '''
                    init_params = {'gammas': gammas.astype(np.float32),
                                   'biases': self.biases,
                                   'weights': self.weights}
            
                    qmc_tim = QMC_TIM_QBM(self.visible, self.hidden, initial_params=init_params, num_replicas=self.population_size, betas=betas, num_its=num_its)
                    self.population = qmc_tim.replicas

                    qmc_tim.train_op(batch, opt)

                    self.biases = qmc_tim.params['biases']
                    self.weights = qmc_tim.params['weights']

                    if debug:
                        '''
                        DEBUG
                        '''
                        bias_nps, weight_nps, bias_pps, weight_pps = self.calculate_exps(betas[-1], batch)
                        print(f'====EXACT====\n\n grads: {bias_pps - bias_nps}, weights: {weight_pps - weight_nps}')
                        #print(f'====EXACT====\n\n grads: bias pps: {bias_pps}, bias_nps: {bias_nps}, weight pps: {weight_pps}, weight nps: {weight_nps} \n\n')
                        qmc_bias_pps, qmc_weight_pps = qmc_tim.positive_phase(data)
                        qmc_bias_nps, qmc_weight_nps = qmc_tim.negative_phase()
                        #print(f'====APPROX=====\n\n grads: bias pps: {qmc_bias_pps}, bias_nps: {qmc_bias_nps}, weight pps: {qmc_weight_pps}, weight nps: {qmc_weight_nps} \n\nEND_ITER\n\n')
                        print(f'====APPROX====\n\n grads: {qmc_bias_pps - qmc_bias_nps}, weights: {qmc_weight_pps - qmc_weight_nps}')

                

                
               

        return pop_history
                
    def sample(self, num_samples):
        if num_samples > self.population_size:
            raise ValueError('Number of samples must be less than population size')
        return self.population[:num_samples]