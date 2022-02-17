import numpy as np
from typing import Tuple

def classical_energy(state: np.ndarray, single_weights: np.ndarray, double_weights: np.ndarray) -> float:
    num_units = state.shape[0]
    E_cl = 0.0
    for unit in range(num_units):
        E_cl -= single_weights[unit] * state[unit]
        for unit2 in range(unit + 1, num_units):
            if (unit, unit2) in double_weights:
                E_cl -= double_weights[(unit, unit2)] * state[unit] * state[unit2]
    return E_cl

def classical_energy_function(state: np.ndarray, beta: float, transverse_field_weight: float, single_weights: np.ndarray, double_weights: np.ndarray) -> float:
    num_units = state.shape[1]
    time_slices = state.shape[0]
    
    E_cl = 0.0
    for m in range(time_slices):
        E_cl += classical_energy(state[m], single_weights, double_weights)
    E_cl /= time_slices

    E_qm = 0.0
    for m in range(time_slices):        
        for a in range(num_units):
            next_time = m + 1
            if next_time == time_slices:
                next_time = 0
            E_qm += np.log(np.tanh(beta * transverse_field_weight[a] / time_slices)) * state[m, a] * state[next_time, a]
    E_qm /= 2 * beta
    
    # print(f'E_qm: {E_qm}')
    # print(f'E_cl: {E_cl}')

    return np.exp(beta * (E_cl + E_qm))

def population_annealing(transverse_field_weight: float, single_weights: np.ndarray, double_weights: np.ndarray, time_slices: int, num_copies: int, init_beta: float, num_steps: int, beta_step_size: float, monte_carlo_iterations: int) -> Tuple[np.ndarray, np.ndarray]:

    num_units = single_weights.shape[0]

    population = np.tile(np.expand_dims(-2 * np.random.uniform(low = 0, high = 2.0, size=(num_copies, num_units)).astype(np.uint32) + np.ones((num_copies, num_units)), axis=1),[1, time_slices, 1])

    history = np.zeros((num_steps, num_copies, time_slices, num_units))

    curr_beta = init_beta
    weights = np.ones(num_copies)

    z_expectations_history = np.zeros((num_steps, num_units))

    M = 2 # what choose? :(

    for i in range(num_steps):

        curr_beta += beta_step_size

        # Calculate relative Boltzmann weights of replicas
        for k in range(num_copies):
            weights[k] = weights[k] * classical_energy_function(population[k], beta_step_size, transverse_field_weight, single_weights, double_weights)
        
        # Resample replicas
        if i % M == 0:
            resample_dist = weights / np.sum(weights)
            #plot_resample(resample_dist)
            for k in range(num_copies):
                population[k] = population[np.random.choice(num_copies, p=resample_dist)]
            weights = np.ones(num_copies)

        # Monte Carlo update
        m_range = np.array(range(time_slices))
        np.random.shuffle(m_range)
        a_range = np.array(range(num_units))
        np.random.shuffle(a_range)
        for k in range(num_copies):
            for m in m_range:
                for a in a_range:
                    for _ in range(monte_carlo_iterations):
                        energy = classical_energy_function(population[k], curr_beta, transverse_field_weight, single_weights, double_weights)

                        # flip state
                        population[k][m][a] = -population[k][m][a]

                        new_energy = classical_energy_function(population[k], curr_beta, transverse_field_weight, single_weights, double_weights)
                        
                        #print(f'energy: {energy}, new_energy: {new_energy}')
                        ratio = new_energy / energy
                        
                        if ratio < np.random.random():
                            # Do not accept transition, flip back
                            population[k][m][a] = -population[k][m][a]
                        

        population_z_vals = np.mean(population, axis=1)
        z_expectations_history[i] = np.average(population_z_vals, axis=0, weights=weights)
    
        history[i] = population

    #plot_dist_with_samples(init_beta + beta_step_size * num_steps, transverse_field_weight, single_weights, double_weights, time_slices, population)

    z_expectations = z_expectations_history[num_steps - 1]

    zz_expectations = {(node1, node2): z_expectations[node1] * z_expectations[node2] for (node1, node2) in double_weights.keys()}
    return z_expectations, zz_expectations, history


def plot_resample(resample_dist):
    import matplotlib.pyplot as plt

    fs = 14
    fig, ax = plt.subplots(figsize=[15,5])
    plt.plot(range(len(resample_dist)), resample_dist, linewidth=3, color='palegreen')
    plt.grid(alpha=.4, linestyle='--')
    plt.xlabel(r'Replica', fontsize=fs)
    plt.ylabel('Probability', fontsize=fs)
    plt.show()
    plt.close()


def plot_dist_with_samples(beta, transverse_field_weight, single_weights, double_weights, time_slices, population):
    import itertools
    import matplotlib.pyplot as plt

    num_qubits = single_weights.shape[0]
    
    probs = np.zeros(2**(num_qubits * time_slices))

    states = np.array(list(itertools.product([0, 1], repeat=(num_qubits * time_slices))))
    states_reshaped = states.reshape((probs.shape[0], time_slices, num_qubits))
    

    expectations = np.zeros(num_qubits)

    for i in range(probs.shape[0]):
        probs[i] = classical_energy_function(states_reshaped[i], beta, transverse_field_weight, single_weights, double_weights)
        z_vals = -np.copy(states_reshaped[i])
        z_vals[z_vals == 0] = 1
        state_z_vals = np.mean(z_vals, axis=0)
        expectations += (state_z_vals * probs[i])
    
    expectations /= np.sum(probs)
    probs /= np.sum(probs)

    population = population.reshape((population.shape[0], -1))
    hist_data = [population[i].dot(2**np.arange(population[i].size)[::-1]) for i in range(population.shape[0])]
    
    print(f'Exact expectations: {expectations}')

    fs = 14
    fig, ax = plt.subplots(figsize=[15,5])
    plt.plot(range(len(probs)), probs, linewidth=3, color='palegreen')
    plt.hist(hist_data, density=True, linewidth=4, color='salmon')
    plt.grid(alpha=.4, linestyle='--')
    plt.xlabel(r'States', fontsize=fs)
    plt.ylabel('Probability', fontsize=fs)
    plt.show()
    plt.close()

    
