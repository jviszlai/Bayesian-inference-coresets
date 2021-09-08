from abc import ABC, abstractmethod
import math, random
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

class AnsatzCircuit(ABC):

    @abstractmethod
    def parse_params(self, param):
        pass

    @abstractmethod
    def construct_V(self, circ, bitstring):
        pass

    @abstractmethod
    def construct_U(self, circ, double_phi, single_phi, barriers=False):
        pass

    @abstractmethod
    def return_ansatz(self, param):
        pass

    @abstractmethod
    def print_params(self, param):
        pass

class HeisenbergAnsatz(AnsatzCircuit):

    def __init__(self, G):
        self.G = G
        self.nq = len(G.nodes())
        self.num_param = 3*self.nq + 3*math.floor(self.nq - 1) + self.nq


    def gen_random_params(self):
        return [random.uniform(0, 2*math.pi) for _ in range(self.num_param)]


    def parse_params(self, param):
        assert (len(param) == self.num_param), 'Incorrect number of parameters'
        one_layer = []
        for i in range(self.nq - 1):
            one_layer.append(param[3*i : 3*(i+1)])
        double_phi = [one_layer]
        one_layer = []
        for i in range(self.nq):
            one_layer.append(param[3*(self.nq-1)+3*i : 3*(self.nq-1)+3*(i+1)])
        single_phi = [one_layer]
        theta = param[-self.nq:]

        return double_phi, single_phi, theta

    def construct_V(self, circ, bitstring):
        for i in range(len(bitstring)):
            if bitstring[i] == 1:
                circ.x(i)

    def construct_U(self, circ, double_phi, single_phi, barriers=False):
        depth = 1
        for j in range(depth):
            for i in range(self.nq-1):
                self._double_rotation(circ, double_phi[j][i], i, i+1)
            if barriers: circ.barrier()
            for i in range(self.nq):
                self._single_rotation(circ, single_phi[j][i], i)
            if barriers: circ.barrier()
            for i in range(self.nq-1):
                self._double_rotation(circ, double_phi[j][i], i, i+1)


    def _single_rotation(self, circ, phi, qubit):
        circ.rz(phi[0], qubit)
        circ.ry(phi[1], qubit)
        circ.rx(phi[2], qubit)


    def _double_rotation(self, circ, phi, q1, q2):
        circ.rzz(phi[0], q1, q2)
        circ.rxx(phi[1], q1, q2)
        self._ryy_gate(circ, q1, q2, phi[2])


    def _ryy_gate(self, circ, q1, q2, theta):
        circ.s([q1, q2])
        circ.rxx(theta, q1, q2)
        circ.sdg([q1, q2])


    def return_ansatz(self, param):
        double_phi, single_phi, theta = self.parse_params(param)

        circ = QuantumCircuit(self.nq)
        #circ.unitary(Gate('V', self.nq, []), list(range(self.nq)))
        self.construct_U(circ, double_phi, single_phi, barriers=True)
        return circ


    def print_params(self, param):
        single_phi, double_phi, theta = self.parse_params(param)

        print('single_phi:', single_phi)
        print('double_phi:', double_phi)
        print('theta:', theta)


class ZZansatz(AnsatzCircuit):

    def __init__(self, G):
        self.G = G
        self.nq = len(G.nodes())
        self.ne = len(G.edges())

        #self.num_param = int(self.nq+(self.nq**2+5*self.nq)/2)
        self.num_param = int(self.nq*(self.nq-1)/2 + 6 + self.nq)


    def gen_random_params(self):
        return [random.uniform(0, 2*math.pi) for _ in range(self.num_param)]


    def parse_params(self, param):
        assert (len(param) == self.num_param), 'Incorrect number of parameters'
        middle_phi = [param[0:self.ne]]
        first_phi = [param[self.ne:self.ne+3]]
        last_phi = [param[self.ne+3:self.ne+6]]
        theta = param[-self.nq:]

        return [first_phi, middle_phi, last_phi, theta]


    def construct_V(self, circ, bitstring):
        """
        Construct the unitary to prepare any computational basis state
        """
        for i in range(len(bitstring)):
            if bitstring[i] == 1:
                circ.x(i)


    def construct_U(self, circ, first_phi, middle_phi, last_phi, barriers=False):
        """
        Construct the ansatz unitary with the form:

            U = Usingle(last_phi)*Udouble(middle_phi)*Usingle(first_phi)
        """
        # We know that the MAXCUT Hamiltonian has ZZ interactions between every pair
        # of qubits so we will implant that structure within our ansatz
        depth = 1 # for now, set the depth to be 1
        for j in range(depth):
            for i in range(self.nq):
                self._singleU(circ, first_phi[j], i)
            for i, edge in enumerate(self.G.edges()):
                self._doubleU(circ, middle_phi[j][i], edge[0], edge[1])
            for i in range(self.nq):
                self._singleU(circ, last_phi[j], i)


    def _singleU(self, circ, phi, qubit):
        """
        Apply single qubit gates to the specified qubit

            Usingle(phi) = exp[i(phi[0]X + phi[1]Y + phi[2]Z)]
        """
        circ.rx(phi[0], qubit)
        circ.ry(phi[1], qubit)
        circ.rz(phi[2], qubit)


    def _doubleU(self, circ, phi, q1, q2):
        """
        Apply a two qubit interaction to the specified pair of qubits

            Udouble(phi) = exp[i*phi*ZZ]
        """
        circ.cx(q1,q2)
        circ.rz(2*phi, q2)
        circ.cx(q1,q2)


    def return_ansatz(self, param):
        first_phi, middle_phi, last_phi, theta = self.parse_params(param)

        circ = QuantumCircuit(self.nq)
        self.construct_U(circ, first_phi, middle_phi, last_phi, barriers=True)
        return circ


    def print_params(self, params):
        first_phi, middle_phi, last_phi, theta = self.parse_params(params)

        print('first_phi:', first_phi)
        print('middle_phi:', middle_phi)
        print('last_phi:', last_phi)
        print('theta:', theta)


















