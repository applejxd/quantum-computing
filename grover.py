# cf. https://dojo.qulacs.org/ja/latest/notebooks/8.2_Grovers_algorithm.html#%E5%AE%9F%E8%A3%85%E4%BE%8B
# cf. http://www.quest.lab.uec.ac.jp/q-school/2010/archive/%E9%87%8F%E5%AD%90%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0.pdf

# ライブラリのインポート
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from qulacs import QuantumState
from qulacs.state import inner_product
from qulacs import QuantumCircuit
from qulacs.gate import to_matrix_gate
from qulacs import QuantumState
from qulacs.gate import Identity, X, Y, Z  # パウリ演算子
from qulacs.gate import H
from qulacs.gate import RX, RY, RZ  # パウリ演算子についての回転演算

matplotlib.use('TkAgg')


# 係数の絶対値の分布をプロットする関数
def show_distribution(state, nqubits):
    plt.bar([i for i in range(pow(2, nqubits))], abs(state.get_vector()))
    plt.show()


def make_Hadamard(nqubits):
    """
    Hadamard's transformation for generating uniform state vector
    cf. https://dojo.qulacs.org/ja/latest/notebooks/3.1_Qulacs_tutorial.html?highlight=add_gate#%E9%87%8F%E5%AD%90%E5%9B%9E%E8%B7%AF%E3%81%AE%E6%A7%8B%E6%88%90

    :param nqubits:
    :return:
    """
    Hadamard = QuantumCircuit(nqubits)
    for i in range(nqubits):
        # tensor product for operators
        Hadamard.add_gate(H(i))
    return Hadamard


def make_U_w(nqubits):
    """
    making Oracle (phase flip w.r.t. answer)
    The answer is |11...1>

    :param nqubits:
    :return:
    """
    U_w = QuantumCircuit(nqubits)
    # Z gate (z-axis Pauli matrix) can be regarded as the phase flip operator
    # apply Z gate to the last qubit ((nqubits-1)-th qubit)
    # cf. http://docs.qulacs.org/ja/latest/intro/4.1_python_tutorial.html#id13
    CnZ = to_matrix_gate(Z(nqubits - 1))
    # phase flip only for |11...1> (phase flip for |1> is sign flip)
    for i in range(nqubits - 1):
        control_index = i
        control_with_value = 1
        # limit operation target for vectors whose i-th element is 1
        # cf. http://docs.qulacs.org/ja/latest/intro/4.1_python_tutorial.html#id12
        CnZ.add_control_qubit(control_index, control_with_value)
    U_w.add_gate(CnZ)
    return U_w


def make_U_s(nqubits):
    """
    Diffusion transformation

    :param nqubits:
    :return:
    """
    U_s = QuantumCircuit(nqubits)
    for i in range(nqubits):
        U_s.add_gate(H(i))

    # 2|0><0| - I = diag(1, -1, -1, ...) の実装
    # まず、位相(-1)を全ての状態に付与する。ゲート行列は array([[-1,0],[0,-1]])
    U_s.add_gate(to_matrix_gate(RZ(nqubits - 1, 2 * np.pi)))

    # ---------------------------- #
    # phase flip only for |00...0> #
    # ---------------------------- #

    # X gate (x axis Pauli's matrix) can be regard as NOT gate
    # |00...00> -> |00...01>
    U_s.add_gate(X(nqubits - 1))
    # |00...01> -> -|00...01>
    CnZ = to_matrix_gate(Z(nqubits - 1))
    for i in range(nqubits - 1):
        control_index = i
        control_with_value = 0
        # limit operation target for vectors whose i-th element is 0
        # cf. http://docs.qulacs.org/ja/latest/intro/4.1_python_tutorial.html#id12
        CnZ.add_control_qubit(control_index, control_with_value)
    U_s.add_gate(CnZ)
    # -|00...01> -> -|00...00>
    U_s.add_gate(X(nqubits - 1))

    for i in range(nqubits):
        U_s.add_gate(H(i))

    return U_s


def main1():
    nqubits = 5

    # 初期状態の準備
    # cf. https://dojo.qulacs.org/ja/latest/notebooks/3.1_Qulacs_tutorial.html?highlight=set_computational_basis#%E9%87%8F%E5%AD%90%E7%8A%B6%E6%85%8B%E3%81%AE%E5%88%9D%E6%9C%9F%E5%8C%96
    state = QuantumState(nqubits)
    state.set_zero_state()
    Hadamard = make_Hadamard(nqubits)
    Hadamard.update_quantum_state(state)

    # check uniform state vector
    show_distribution(state, nqubits)

    # check phase flip w.r.t. answer
    hoge = state.copy()
    U_w = make_U_w(nqubits)
    U_w.update_quantum_state(hoge)
    print(hoge.get_vector())

    # U_s U_w を作用
    U_s = make_U_s(nqubits)
    U_w.update_quantum_state(state)
    U_s.update_quantum_state(state)
    show_distribution(state, nqubits)

    # 内積を評価するために 解状態 |1...1> を作っておく
    target_state = QuantumState(nqubits)
    # 2**n_qubits-1 は 2進数で 1...1
    target_state.set_computational_basis(2 ** nqubits - 1)

    # グローバーのアルゴリズムの実行
    state = QuantumState(nqubits)
    state.set_zero_state()

    # generate uniform state vector
    Hadamard.update_quantum_state(state)

    for i in range(4):
        U_w.update_quantum_state(state)
        U_s.update_quantum_state(state)
        show_distribution(state, nqubits)
        print(np.linalg.norm(inner_product(state, target_state)))


def main2():
    nqubits = 10
    state = QuantumState(nqubits)
    state.set_zero_state()

    # 内積を評価するために 解状態 |1...1> を作っておく
    target_state = QuantumState(nqubits)
    # 2**n_qubits-1 は 2進数で 1...1
    target_state.set_computational_basis(2 ** nqubits - 1)

    # グローバーのアルゴリズムの実行
    Hadamard = make_Hadamard(nqubits)
    U_w = make_U_w(nqubits)
    U_s = make_U_s(nqubits)

    result = []

    state = QuantumState(nqubits)
    state.set_zero_state()
    Hadamard.update_quantum_state(state)
    for k in range(30):
        U_w.update_quantum_state(state)
        U_s.update_quantum_state(state)
        # show_distribution(state,nqubits)
        result.append(np.linalg.norm(inner_product(state, target_state)))

    max_k = np.argmax(result)
    print(f"maximal probability {result[max_k]:5e} is obtained at k = {max_k + 1}")

    plt.plot(np.arange(1, 30 + 1), result, "o-")
    plt.show()


def main3():
    result = []
    min_nqubits = 6
    max_nqubits = 16
    for nqubits in range(min_nqubits, max_nqubits + 1, 2):
        # 回路の準備
        Hadamard = make_Hadamard(nqubits)
        U_w = make_U_w(nqubits)
        U_s = make_U_s(nqubits)

        # 内積を評価するために 解状態 |1...1> を作っておく
        target_state = QuantumState(nqubits)
        # 2**n_qubits-1 は 2進数で 1...1
        target_state.set_computational_basis(2 ** nqubits - 1)

        state = QuantumState(nqubits)
        state.set_zero_state()
        Hadamard.update_quantum_state(state)

        # 確率が減少を始めるまで U_s U_w をかける
        tmp = 0
        flag = 0
        num_iter = 0
        while flag == 0 and num_iter <= 1000:
            num_iter += 1
            U_w.update_quantum_state(state)
            U_s.update_quantum_state(state)
            suc_prob = np.linalg.norm(inner_product(state, target_state))
            if tmp < suc_prob:
                tmp = suc_prob
            else:
                flag = 1
        result.append([nqubits, num_iter, suc_prob])
        print(f"nqubits={nqubits}, num_iter={num_iter}, suc_prob={suc_prob:5e}")

        result_array = np.array(result)

    plt.xlim(min_nqubits - 1, max_nqubits + 1)
    plt.xlabel("n, # of qubits", fontsize=15)
    plt.ylabel("k, # of iteration", fontsize=15)
    plt.semilogy(result_array[:, 0], result_array[:, 1], "o-", label="experiment")
    plt.semilogy(result_array[:, 0], 0.05 * 2 ** result_array[:, 0], "-", label=r"$\propto N=2^n$")
    plt.semilogy(result_array[:, 0], 2 ** (0.5 * result_array[:, 0]), "-", label=r"$\propto \sqrt{N}=2^{n/2}$")
    plt.legend(fontsize=10)
    plt.show()


if __name__ == "__main__":
    main1()
    main2()
    main3()
