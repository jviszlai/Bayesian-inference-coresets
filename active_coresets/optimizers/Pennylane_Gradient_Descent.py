import random as rand
import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Tensor


def pennylane_maxcut_circuit(P, gamma, beta, G):
    return None


def get_maxcut_Hamiltonian(G):
    coefs, operators = [], []
    for edge in G.edges():
        new_op = [qml.Identity(i) for i in range(len(G.nodes()))]
        coefs.append(G[edge[0]][edge[1]]['weight'])
        new_op[edge[0]] = qml.PauliZ(edge[0])
        new_op[edge[1]] = qml.PauliZ(edge[1])
        operators.append(Tensor(*new_op))
    hamiltonian = qml.Hamiltonian(coefs, operators)
    return hamiltonian


def gradient_descent(G, P, optimizer='default', stepsize=0.01, steps=100,
                     init_params=None, verbose=1):
    nq = len(G.nodes)

    # create the Maxcut Hamiltonian
    hamiltonian = get_maxcut_Hamiltonian(G)

    # create the device
    dev = qml.device("default.qubit", wires=nq, analytic=True, shots=1)

    def ansatz(params, **kwargs):
        # parse the parameters
        gammas = [params[i] for i in range(len(params)) if i % 2 == 0]
        betas  = [params[i] for i in range(len(params)) if i % 2 == 1]

        # Hadamard layer
        for q in range(nq):
            qml.Hadamard(q)

        # alternate between the cost and mixer unitaries p times
        for p in range(P):
            # evolve under the cost Hamiltonian
            for edge in G.edges():
                i, j = edge
                qml.CNOT(wires=[i,j])
                qml.RZ(2 * G[i][j]['weight'] * gammas[p], wires=j)
                qml.CNOT(wires=[i,j])

            # evolve under the mixer Hamiltonian
            for q in range(nq):
                qml.RX(2 * betas[p], wires=q)

    def cost(params):
        # Pennylane optimizers MINIMIZE the objective function
        # so we return the negative of the computed energy
        return -1 * qml.VQECost(ansatz, hamiltonian, dev)(params)

    # initialize the optimizer
    if optimizer is 'default':
        opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    elif optimizer is 'ada':
        opt = qml.AdagradOptimizer(stepsize=stepsize)
    elif optimizer is 'adam':
        opt = qml.AdamOptimizer(stepsize=stepsize, beta1=0.5, beta2=0.9)
    elif optimizer is 'qng':
        opt = qml.QNGOptimizer(stepsize=stepsize)
    else:
        raise Exception('Optimizer: {} is not supported'.format(optimizer))

    # set initial parameter values
    if init_params is None:
        params = [rand.uniform(-2*np.pi,2*np.pi) for i in range(2*P)]
    else:
        params = init_params

    # begin the optimization
    for i in range(steps):
        # update the circuit parameters
        params = opt.step(cost, params)

        if verbose > 0 and (i + 1) % 10 == 0:
            print('Cost after step {:5d}: {:.7f}'.format(i+1, cost(params)))

    return params, cost(params)
