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

    def compare(self, qmc_tim):
        num_qubits = len(self.visible) + len(self.hidden)
        H = np.zeros((2 ** num_qubits, 2 ** num_qubits))

        pauli_z = np.array([[1, 0], [0, -1]])
        pauli_x = np.array([[0, 1], [1, 0]])
        identity = np.array([[1, 0], [0, 1]])

        for i, b in enumerate(self.biases):
            m = b
            for j in range(num_qubits):
                if j == i:
                    m = np.kron(m, pauli_z)
                else:
                    m = np.kron(m, identity)
            H = np.subtract(H, m)

        for a in range(len(self.visible)):
            for b in range(len(self.hidden)):
                m = self.weights[a][b]
                for j in range(num_qubits):
                    if j == a or j == b:
                        m = np.kron(m, pauli_z)
                    else:
                        m = np.kron(m, identity)
                H = np.subtract(H, m)
        
        for i in range(num_qubits):
            m = self.gamma
            for j in range(num_qubits):
                if j == i:
                    m = np.kron(m, pauli_x)
                else:
                    m = np.kron(m, identity)
            H = np.subtract(H, m)
        
        beta = 1
        H = -1 * beta * H
        exp_H = scipy.linalg.expm(H)
        partition_function = np.trace(exp_H)
        exact_thermal_state = qiskit.quantum_info.DensityMatrix(exp_H / partition_function)

        expectations = np.zeros(num_qubits)

        for i in range(num_qubits):
            pauli_str = 'I' * (i) + 'Z' + 'I' * (num_qubits - i - 1)
            expectations[i] = exact_thermal_state.expectation_value(PauliSumOp.from_list([(pauli_str, 1)])).real

        print(f'Exact expectations: {expectations}')

        z_exp, zz_exp = qmc_tim.negative_phase()
        print(f'Approx expectations: z: {z_exp}')

    def train(self, data: np.ndarray, epochs: int, batch_size, betas: np.ndarray, num_its: int = 10, learning_rate: float = 0.001):
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
                print(f'...Batch {i // batch_size + 1} / {data.shape[0] // batch_size}')
                
                init_params = {'gammas': gammas.astype(np.float32),
                               'biases': self.biases,
                               'weights': self.weights}

                print(init_params)

                qmc_tim = QMC_TIM_QBM(self.visible, self.hidden, initial_params=init_params, num_replicas=self.population_size, betas=betas, num_its=num_its)
                self.population = qmc_tim.replicas

                self.compare(qmc_tim)

                pop_history.append(self.population)

                batch = data[i:i + batch_size]
                qmc_tim.train_op(batch, opt)
                
                self.biases = qmc_tim.params['biases']
                self.weights = qmc_tim.params['weights']

        return pop_history
                
    def sample(self, num_samples):
        if num_samples > self.population_size:
            raise ValueError('Number of samples must be less than population size')
        return self.population[:num_samples]