# Neural-Network-Algorithm

## Files Introduction

This repository contains two Python scripts that interact with distinct neural network algorithms located in the Model package:

<dl>
    <dt><b>manual_nn.py</b></dt>
    <dd>This script executes a neural network algorithm manually implemented using only the numpy library. The implementation is housed in the `NeuralNetwork` module within the Model package. Running `manual_nn.py` trains and tests a neural network model using this manual implementation.</dd>
    <dt><b>auto_nn.py</b></dt>
    <dd>This script utilizes the PyTorch machine learning library to automate forward and backward propagation processes. The implementation can be found in the `NeuralNetworkAutoDifferentiation` module of the Model package. It allows for performance comparison across different activation functions and hidden layer configurations.</dd>
</dl>

Both Neural Network Algorithms are online learning.

## Getting Started
Follow the steps below to set up and run the neural network algorithms:

<ol type="I">
    <li>Fork this repository to your GitHub account and clone it to your local machine.</li>
    <li>Open a terminal and navigate to the directory where you cloned the repository.</li>
    <li>Execute the following commands in your terminal:</li>
        <ol type="i">
            <li>Create a virtual environment: `python3 -m venv myenv`</li>
            <li>Activate the virtual environment: </li>
                <ol type="i">
                    <li>Windows command: `myenv/bin/activate`</li>
                    <li>Linux or macOS command: `source myenv/bin/activate`</li>
                </ol>
            <li>Upgrade pip: `pip install --upgrade pip`</li>
            <li>Install required packages: `pip install -r requirements.txt`</li>
            <li>Run the manual neural network script: `python3 manual_nn.py`</li>
            <li>Run the automatic neural network script: `python3 auto_nn.py`</li>
            <li>Deactivate the virtual environment: `deactivate`</li>
        </ol>
</ol>