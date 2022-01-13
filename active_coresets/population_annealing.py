import numpy as np

def classical_energy(state: np.ndarray, single_weights: np.ndarray, double_weights: np.ndarray) -> float:
    num_units = state.shape[0]
    E_cl = 0.0
    for unit in range(num_units):
        if state[unit] == 1:
            E_cl -= single_weights[unit]
        for unit2 in range(unit + 1, num_units):
            if state[unit] == 1 and state[unit2] == 1 and (unit, unit2) in double_weights:
                E_cl -= double_weights[(unit, unit2)]
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
            if state[m, a] == 1 and state[next_time, a] == 1:
                E_qm += np.log(np.tanh(beta * transverse_field_weight / time_slices))
    E_qm /= 2 * beta

    return np.exp(-beta * (E_cl + E_qm))

def population_annealing(transverse_field_weight: float, single_weights: np.ndarray, double_weights: np.ndarray, time_slices: int, num_copies: int, init_beta: float, num_steps: int, beta_step_size: float, monte_carlo_iterations: int) -> np.ndarray:
    num_units = single_weights.shape[0]

    population = np.random.randint(0, 2, size=(num_copies, time_slices, num_units))

    history = np.zeros((num_steps, num_copies, time_slices, num_units))

    expectations = np.zeros((num_steps, num_units))

    curr_beta = init_beta
    weights = np.ones(num_copies)

    M = 1 # what choose? :(

    for i in range(num_steps):
        
        history[i] = population

        # Calculate relative Boltzmann weights of replicas
        for k in range(num_copies):
            weights[k] = weights[k] * classical_energy_function(population[k], beta_step_size, transverse_field_weight, single_weights, double_weights)
        
        # Resample replicas
        if i % M == 0:
            resample_dist = weights / np.sum(weights)
            plot_resample(resample_dist)
            for k in range(num_copies):
                population[k] = population[np.random.choice(num_copies, p=resample_dist)]
            weights = np.ones(num_copies)

        # Monte Carlo update
        for k in range(num_copies):
            
            for _ in range(monte_carlo_iterations):
                energy = classical_energy_function(population[k], curr_beta, transverse_field_weight, single_weights, double_weights)

                # Randomly flip one state
                transition_idx = (np.random.randint(0, time_slices), np.random.randint(0, num_units))
                population[k][transition_idx] = 1 - population[k][transition_idx]

                new_energy = classical_energy_function(population[k], curr_beta, transverse_field_weight, single_weights, double_weights)

                ratio = new_energy / energy
                
                if ratio < np.random.random():
                    # Do not accept transition, flip back
                    population[k][transition_idx] = 1 - population[k][transition_idx]

        curr_beta += beta_step_size

    
        
    z_expectations = -np.copy(population)
    z_expectations[z_expectations == 0] = 1

    return np.mean(z_expectations, axis=(0, 1)), history


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