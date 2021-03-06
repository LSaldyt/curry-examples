from pyquil import Program, get_qc
from pyquil.gates import *

# construct a Bell State program
p = Program(H(0), CNOT(0, 1))
# run the program on a QVM
qvm = get_qc('9q-square-qvm')
result = qvm.run_and_measure(p, trials=10)
print(result)


