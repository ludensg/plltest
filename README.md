# Distributed TensorFlow Training Project

This project aims to compare the performance of data and model parallelism in distributed deep learning using TensorFlow.

## Overview

The project consists of training scripts and server scripts to facilitate distributed training across multiple machines. The main objective is to evaluate the efficiency and performance of data parallelism and model parallelism in a distributed environment.

## System Configuration

The project is designed to run on the following machines:

- **ns31**:
  - Role: Parameter Server
  - CPU: Intel(R) Xeon(R) Gold 6240 @ 2.60GHz
  - Cores: 36
  - Memory: 187G
  - OS: CentOS Linux 7 (Core)

- **inv01**:
  - Role: Worker 0
  - CPU: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz
  - Cores: 40
  - Memory: 62G
  - OS: CentOS Linux 7 (Core)

- **inv02**:
  - Role: Worker 1
  - CPU: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz
  - Cores: 40
  - Memory: 62G
  - OS: CentOS Linux 7 (Core)

## Scripts

- `ps_server.py`: Starts the TensorFlow server on `ns31` as the parameter server.
- `worker0_server.py`: Starts the TensorFlow server on `inv01` as Worker 0.
- `worker1_server.py`: Starts the TensorFlow server on `inv02` as Worker 1.
- `exp.py`: Main training script that conducts both data and model parallelism training and plots the results.

## Setup and Execution

1. **Start TensorFlow Servers**:
   - On `ns31`, run: `python ps_server.py`
   - On `inv01`, run: `python worker0_server.py`
   - On `inv02`, run: `python worker1_server.py`

2. **Run the Main Training Script**:
   - On `ns31`, run: `python exp.py`

3. **Monitor Progress**:
   The scripts provide print updates on the status of the servers and the progress of the training.

4. **Results**:
   After training, a comparison graph (`comparison_graph.png`) will be generated,
