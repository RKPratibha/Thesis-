# Thesis-

# Accelerating-Parameter-Optimization-for-Quantum-QML-using-FPGA
Accelerating Parameter Optimization for Quantum Machine Learning (QML) using Field Programmable Gate Arrays (FPGA)
This repository contains an implementation of a Quantum Convolutional Neural Network (QCNN) with FPGA acceleration for the gradient computation and optimization steps. The model demonstrates how hardware acceleration can be applied to quantum machine learning to achieve significant performance improvements.

## Overview

This project combines quantum computing and FPGA hardware acceleration to create an efficient implementation of a quantum neural network for image classification. The model is designed to classify simple patterns (horizontal vs. vertical lines) using a quantum circuit with up to 12 qubits.

### Key Features

- Hybrid quantum-classical architecture with FPGA acceleration
- Parameterized quantum circuit using Qiskit
- Custom FPGA kernel for gradient computation and weight updates
- Comprehensive performance analysis comparing CPU and FPGA implementations
- Detailed timing metrics for optimization and profiling

## Architecture

The system uses a hybrid architecture:

1. **CPU (Python/Qiskit)**: Handles quantum circuit simulation, data preprocessing, and model management
2. **FPGA (HLS C++)**: Accelerates the computationally intensive gradient calculation and weight updates

### Quantum Circuit Design

The quantum circuit consists of:

- **Input Encoding Layer**: Hadamard gates followed by Ry rotations to encode classical data
- **Variational Layers**: Multiple layers of rotation gates (Rx, Ry, Rz) and CNOT entanglements
- **Measurement**: Final state vector is used to compute class probabilities

The circuit uses 8 qubits by default but has been tested with 4-12 qubits.

### FPGA Acceleration

The FPGA kernel implements:

- Cross-entropy loss calculation
- Numerical gradient computation using finite difference method
- Weight updates using gradient descent

The kernel is optimized with:
- Parallel processing of multiple weights
- Block-based memory access patterns
- Local memory caching for reduced latency
- HLS pragmas for efficient hardware synthesis

## Performance Results

Performance comparison between CPU and FPGA implementations:

| Qubits | CPU Time (s) | CPU Accuracy | FPGA Time (s) | FPGA Accuracy | Speedup |
|--------|--------------|--------------|---------------|---------------|---------|
| 4      | 2.85         | 95%          | 1.19          | 95%           | 2.39x   |
| 5      | 3.25         | 95%          | 1.32          | 95%           | 2.46x   |
| 8      | 3.48         | 90%          | 1.39          | 85%           | 2.50x   |
| 10     | 6.05         | 90%          | 1.40          | 85%           | 4.32x   |
| 12     | 7.03         | 90%          | 1.53          | 85%           | 4.59x   |

The results show that:
1. FPGA acceleration provides significant speedup, especially for larger qubit counts
2. The speedup factor increases with model complexity, reaching 4.59x for 12 qubits
3. Both implementations maintain comparable accuracy levels

## Implementation Details

### CPU Implementation

The Python implementation includes:

- Data generation with controlled noise for horizontal and vertical patterns
- Parameterized quantum circuit construction using Qiskit
- Forward pass computation using statevector simulation
- Mini-batch training with detailed timing metrics

### FPGA Kernel

The HLS C++ kernel (`Optimizer_krnl.cpp`) implements:

- Memory-efficient block processing with local buffers
- Parallel computation of gradients for multiple parameters
- Hardware-optimized numerical operations using HLS libraries
- AXI interfaces for high-bandwidth data transfer

### Integration

The CPU and FPGA components communicate via:

- PYNQ framework for Python-FPGA integration
- Memory buffers for efficient data transfer
- Synchronization mechanisms to coordinate computation

## Dependencies

- Python 3.7+
- Qiskit
- NumPy, Matplotlib, scikit-learn
- PYNQ
- Vitis HLS (for FPGA kernel compilation)
- Xilinx FPGA hardware (U200 board used in testing)

## Usage

1. Compile the FPGA kernel:
```bash
v++ -c -t hw --platform xilinx_u200_gen3x16_xdma_2_202110_1 -k Optimizer_krnl Optimizer_krnl.cpp -o Optimizer_krnl.xo
v++ -l -t hw --platform xilinx_u200_gen3x16_xdma_2_202110_1 --config kernel.cfg Optimizer_krnl.xo -o Optimizer_krnl.xclbin
```

2. Run the QCNN model:
```python
from qcnn_fpga import train_quantum_circuit

weights, circuit, input_params, weight_params = train_quantum_circuit(
    n_samples=100,
    n_epochs=50,
    batch_size=32,
    learning_rate=0.01
)
```




```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
