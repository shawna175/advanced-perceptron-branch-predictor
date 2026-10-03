# Advanced Perceptron-Based Dynamic Branch Predictor

A configurable **dynamic branch prediction simulator** implemented in C using the Perceptron Learning Algorithm. The project simulates branch prediction using global branch history, adaptive weight training, and multiple synthetic or file-based branch traces.

## Overview

Branch prediction is an important technique in modern computer architecture that helps reduce pipeline stalls caused by conditional branches.

This project implements a perceptron-based dynamic branch predictor that learns from previous branch outcomes and uses weighted history information to predict future branches. It provides an interactive simulation environment for exploring different predictor configurations and evaluating prediction performance.

## Key Features

- **Perceptron-Based Prediction:** Uses a weighted sum of global branch history and a bias weight to predict branch outcomes.
- **Adaptive Training:** Updates weights when a prediction is incorrect or the prediction confidence falls below a predefined threshold.
- **Saturated Weight Updates:** Prevents weights from exceeding their configured bit-width limits.
- **Global History Register (GHR):** Maintains recent branch outcomes for history-based prediction.
- **PC Hashing:** Maps program counter addresses to perceptron table entries.
- **Multiple Branch Patterns:** Supports loop, alternating, XOR, correlated, random, and mixed synthetic patterns.
- **Trace File Support:** Allows simulations using externally supplied branch traces.
- **Configurable Parameters:** Supports custom table size, history length, weight bits, simulation steps, and reporting intervals.
- **Performance Evaluation:** Reports prediction accuracy, mispredictions, training updates, execution time, and approximate predictions per second.

## Prediction Mechanism

The predictor calculates a weighted sum of the bias and global branch history:

\[
Y = w_0 + \sum_{i=1}^{m} w_i x_i
\]

Where:

- `w0` is the bias weight.
- `wi` represents the weight associated with the `i`-th history bit.
- `xi ∈ {-1, +1}` represents the corresponding branch history outcome.
- `m` is the configured history length.

**Prediction rule:**

- If `Y >= 0`, predict Taken (`+1`).
- Otherwise, predict Not Taken (`-1`).

**Training condition:**

Training occurs when the prediction is incorrect or when the absolute prediction sum is less than or equal to the training threshold.

\[
\theta = \lfloor 1.93m \rfloor + 14
\]

The weights are updated using the actual branch outcome and the corresponding history values, with saturation to maintain the configured weight range.

## Technologies

- **Language:** C
- **Compiler:** GCC / Clang / MinGW
- **Core Concepts:** Computer Architecture, Dynamic Branch Prediction, Perceptron Learning, Global History Register, Performance Simulation

## Getting Started

### Prerequisites

- GCC or Clang compiler
- Terminal or command-line environment
- Git (optional, for cloning the repository)

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/advanced-perceptron-branch-predictor.git
cd advanced-perceptron-branch-predictor
```

Replace `YOUR_USERNAME` with your GitHub username.

### Compile

```bash
gcc -O2 -Wall -Wextra -std=c11 Implement_a_dynamic_branch_predictorF.c -o branch_predictor
```

### Run

```bash
./branch_predictor
```

On Windows:

```bash
branch_predictor.exe
```

The program interactively prompts for the simulation configuration, including:

| Parameter | Default | Description |
|---|---:|---|
| Table Size | 4096 | Number of perceptron entries; must be a power of two |
| History Length | 32 | Number of global branch history bits |
| Weight Bits | 8 | Signed weight precision |
| Simulation Steps | 200,000 | Maximum number of simulated branch outcomes |
| Reporting Interval | 10,000 | Frequency of intermediate performance summaries |
| Pattern | mixed | Synthetic branch pattern used for simulation |

Press Enter to use the default value for each parameter.

## Supported Branch Patterns

| Pattern | Description |
|---|---|
| `loop` | Simulates mostly taken branches with periodic not-taken outcomes |
| `alternating` | Generates alternating taken and not-taken outcomes |
| `xor` | Generates outcomes based on the parity of the combined PC and GHR bits |
| `correlated` | Generates outcomes based on a selected global history bit |
| `random` | Generates random taken and not-taken outcomes |
| `mixed` | Cycles through different synthetic branch behaviors |

These patterns provide different simulation scenarios for examining how the predictor responds to structured, history-dependent, and random branch behavior.

## Trace File Format

The simulator also supports reading branch traces from a file.

Each valid line contains a program counter followed by a branch outcome:

```text
0x400100 1
0x400104 0
0x400108 1
0x40010C 1
0x400110 0
```

- The first value is the program counter (PC), which can be specified in hexadecimal or another format supported by the C integer parser.
- The second value represents the actual branch outcome: `1` for Taken and `0` for Not Taken.
- Lines beginning with `#` and blank lines are ignored.

Select trace file mode when prompted and provide the path to the trace file.

## Performance Metrics

The simulator reports the following metrics:

| Metric | Description |
|---|---|
| Prediction Accuracy | Percentage of correctly predicted branches |
| Correct Predictions | Total number of correct predictions |
| Mispredictions | Total number of incorrect predictions |
| Trained Updates | Number of training operations performed |
| Low-Confidence Updates | Number of updates triggered by the confidence threshold |
| Elapsed CPU Time | CPU time used for the simulation |
| Predictions per Second | Approximate prediction throughput |

Prediction accuracy is calculated as:

\[
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100
\]

Actual results depend on the selected configuration, branch pattern, trace input, and random initialization.

## Project Structure

```text
advanced-perceptron-branch-predictor/
│
├── Implement_a_dynamic_branch_predictorF.c
├── README.md
└── Project_Report.pdf
```

The repository can be extended with separate source, documentation, test, trace, and results directories as the project evolves.

## Future Improvements

- Implement baseline predictors such as Bimodal and GShare for comparative evaluation.
- Introduce configurable random seeds for reproducible experiments.
- Analyze prediction accuracy across different history lengths and table sizes.
- Add automated testing and stronger input validation.
- Generate CSV-based experimental results and performance visualizations.
- Explore hybrid or gated predictor architectures.
- Investigate fixed-point optimizations and hardware-oriented implementation constraints.

## Academic Context

**Course:** CSE 360 – Computer Architecture  
**Department:** Computer Science and Engineering  
**Institution:** East West University

This project was developed as part of an academic course project to explore dynamic branch prediction, perceptron learning, and performance simulation in computer architecture.

## License

This project is available for educational and research purposes. A formal license can be added after deciding the preferred licensing terms.
