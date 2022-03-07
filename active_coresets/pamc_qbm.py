import numpy as np
import tensorflow as tf

from active_coresets.qmc_tim_qbm import QMC_TIM_QBM

class PAMC_QBM():

    def __init__(self, visible: np.ndarray, hidden: np.ndarray, gamma: float, population_size: int):
        self.visible = visible
        self.hidden = hidden
        self.biases = np.random.normal(scale=0.5, size=(len(visible) + len(hidden))).astype(np.float32)
        self.weights = np.random.normal(scale=0.01, size=(len(visible), len(hidden))).astype(np.float32)
        self.gamma = gamma
        self.population_size = population_size
        self.population = -2 * np.random.randint(0, 2, size=(population_size, len(visible) + len(hidden))) + np.ones((population_size, len(visible) + len(hidden)))

    def train(self, data: np.ndarray, epochs: int, batch_size: int, betas: np.ndarray, num_its: int):
        pop_history = [self.population]

        gammas = np.ones(len(self.visible) + len(self.hidden)) * self.gamma
        init_params = {'gammas': gammas.astype(np.float32),
                       'biases': self.biases,
                       'weights': self.weights}

        qmc_tim = QMC_TIM_QBM(self.visible, self.hidden, initial_params=init_params, num_replicas=self.population_size, betas=betas, num_its=num_its, init_compute=False)

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
            for i in range(0, data.shape[0], batch_size):
                print(f'...Batch {i // batch_size + 1} / {data.shape[0] // batch_size}')
                self.population = qmc_tim.pa()[:, 0]
                qmc_tim.replicas = self.population

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