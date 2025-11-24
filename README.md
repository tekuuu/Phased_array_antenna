# Microstrip Linear Array Simulator

This project provides an interactive simulator for microstrip linear antenna arrays, focusing on X-band frequencies. The simulator allows users to visualize and analyze the radiation pattern, side lobe levels (SLL), gain, bandwidth, and other key parameters of microstrip patch arrays.

## Features

- Adjustable number of elements, phase per element, and window type (uniform, Dolph-Chebyshev, binomial, hamming)
- Realistic modeling of microstrip patch element patterns and input impedance
- Visualization of radiation pattern and amplitude distribution
- Calculation of array parameters including gain, SLL, HPBW, efficiency, input impedance, VSWR, return loss, and bandwidth

## Usage

Run the simulator with:

```sh
python linear.py
```

An interactive window will appear, allowing you to adjust array parameters and observe their effects in real time.

## Requirements

- Python 3.x
- numpy
- scipy
- matplotlib

Install dependencies with:

```sh
pip install numpy scipy matplotlib
```

## License

This project is provided for educational purpose
