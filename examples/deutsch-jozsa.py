import itertools
import numpy as np
import pyquil.api as api
from pyquil.gates import *
from pyquil.quil import Program

def qubit_strings(n):
    qubit_strings = []
    for q in itertools.product(['0', '1'], repeat=n):
        qubit_strings.append(''.join(q))
    return qubit_strings

def black_box_map(n, balanced):
    """
    Black-box map, f(x), on n qubits represented by the vector x
    """
    qubs = qubit_strings(n)

    # assign a constant value to all inputs if not balanced
    if not balanced:
        const_value = np.random.choice([0, 1])
        d_blackbox = {q: const_value for q in qubs}

    # assign 0 to half the inputs, and 1 to the other inputs if balanced
    if balanced:
        # randomly pick half the inputs
        half_inputs = np.random.choice(qubs, size=int(len(qubs)/2), replace=False)
        d_blackbox = {q_half: 0 for q_half in half_inputs}
        d_blackbox_other_half = {q_other_half: 1 for q_other_half in set(qubs) - set(half_inputs)}
        d_blackbox.update(d_blackbox_other_half)

    return d_blackbox

def qubit_ket(qub_string):
    """
    Form a basis ket out of n-bit string specified by the input 'qub_string', e.g.
    '001' -> |001>
    """
    e0 = np.array([[1], [0]])
    e1 = np.array([[0], [1]])
    d_qubstring = {'0': e0, '1': e1}

    # initialize ket
    ket = d_qubstring[qub_string[0]]
    for i in range(1, len(qub_string)):
        ket = np.kron(ket, d_qubstring[qub_string[i]])

    return ket

def projection_op(qub_string):
    """
    Creates a projection operator out of the basis element specified by 'qub_string', e.g.
    '101' -> |101> <101|
    """
    ket = qubit_ket(qub_string)
    bra = np.transpose(ket)  # all entries real, so no complex conjugation necessary
    proj = np.kron(ket, bra)
    return proj

def black_box(n, balanced):
    """
    Unitary representation of the black-box operator on (n+1)-qubits
    """
    d_bb = black_box_map(n, balanced=balanced)
    # initialize unitary matrix
    N = 2**(n+1)
    unitary_rep = np.zeros(shape=(N, N))
    # populate unitary matrix
    for k, v in d_bb.items():
        unitary_rep += np.kron(projection_op(k), np.eye(2) + v*(-np.eye(2) + np.array([[0, 1], [1, 0]])))

    return unitary_rep

p = Program()

# pick number of control qubits to be used
n = 5

# Define U_f
p.defgate("U_f", black_box(n, balanced=False))

# Prepare the starting state |0>^(\otimes n) x (1/sqrt[2])*(|0> - |1>)
for n_ in range(1, n+1):
    p.inst(I(n_))
p.inst(X(0))
p.inst(H(0))

# Apply H^(\otimes n)
for n_ in range(1, n+1):
    p.inst(H(n_))

# Apply U_f
p.inst(("U_f",) + tuple(range(n+1)[::-1]))

# Apply final H^(\otimes n)
for n_ in range(1, n+1):
    p.inst(H(n_))

# Final measurement
classical_regs = list(range(n))
for i, n_ in enumerate(list(range(1, n+1))[::-1]):
    p.measure(n_, classical_regs[i])

qvm = api.QVMConnection()
measure_n_qubits = qvm.run(p, classical_regs)

# flatten out list
measure_n_qubits = [item for sublist in measure_n_qubits for item in sublist]

# Determine if function is balanced or not
if set(measure_n_qubits) == set([0, 1]):
    print ("Function is balanced")
elif set(measure_n_qubits) == set([0]):
    print ("Function is constant")


