import qiskit
import cuquantum


def main():
    qc = qiskit.QuantumCircuit(1)
    qc.x(0)
    converter = cuquantum.CircuitToEinsum(qc)
    expr, operands = converter.amplitude(bitstring="1")
    print(expr)
    for op in operands:
        print(op)
        print()
    print(cuquantum.contract(expr, *operands))


if __name__ == "__main__":
    main()
