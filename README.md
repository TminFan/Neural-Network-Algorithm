# Neural-Network-Algorithm

## Files Introduction

This repository contains two Python scripts that interact with distinct neural network algorithms located in the Model package. Both Neural Network Algorithms are online learning.:

<dl>
    <dt><b>manual_nn.py</b></dt>
    <dd>This script executes a neural network algorithm manually implemented using only the numpy library. The implementation is housed in the `NeuralNetwork` module within the Model package. Running `manual_nn.py` trains and tests a neural network model using this manual implementation.</dd>
    <dt><b>auto_nn.py</b></dt>
    <dd>This script utilizes the PyTorch machine learning library to automate forward and backward propagation processes. The implementation can be found in the `NeuralNetworkAutoDifferentiation` module of the Model package. It allows for performance comparison across different activation functions and hidden layer configurations.</dd>
</dl>

<p>
A GitHub workflow has been added to automatically test model development in `auto_nn.py`. This workflow is triggered by any `push` event.
</p>
<h3>Workflow Outputs:</h3>
<dl>
    <dt><b>Model Performance:</b></dt>
    <dd>Performance metrics are captured in the `report.html.jinja` template and displayed in the GitHub job step summary upon workflow completion.</dd>
    <dt><b>Training History Plots:</b></dt>
    <dd>All plots showing model training history are uploaded as artifacts and can be accessed on the same page as the workflow job summary.</dd>
</dl>

<h3>Caching Dependencies:</h3>
<p>
The workflow leverages GitHub's cache action to speed up the pipeline by caching environment dependencies. The initial run of the workflow takes approximately six minutes. Subsequent runs are faster, approximately two minutes, due to dependency caching. Note that there are some limitations to using GitHub's cache action. For more details, refer to the <a href='https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/caching-dependencies-to-speed-up-workflows#usage-limits-and-eviction-policy'>"Cache dependencies".</a>
</p>

<h3>Local Testing:</h3>
The provided Dockerfile facilitates testing the model on local machines.

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