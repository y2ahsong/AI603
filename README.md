# AI603 - Problem Sets 4
20245360 Yeseong Jung

## Problem 1
It is based on the [[Neural Flows Experiments](https://github.com/mbilos/neural-flows-experiments)]. It extends the functionality by implementing and comparing additional flow models, including MLP Flow, alongside existing models such as ResNet Flow, GRU Flow, and Coupling Flow. The goal is to evaluate the performance of these models on key metrics, including test loss, extrapolation capabilities, and training speed using synthetic ellipse dataset.

### Settings
- Dataset
    - The ellipse dataset from the synthetic experiment was used for all comparisons.
- Models
    - ResNet Flow
	- Coupling Flow
	- GRU Flow
    - MLP Flow: A flow model based on multi-layer perceptrons, created as part of this experiment. The MLP Flow model is implemented using a series of linear layers with ReLU activations.
- Evaluation Metrics
    - test_loss: Mean loss on the test dataset
    - loss_extrap_time: Loss from temporal extrapolation
	- loss_extrap_space: Loss from spatial extrapolation
	- epoch_duration_mean: Average training time per epoch
- Training Details
	- Solver: dopri5
	- Optimizer: Adam

### Implementation
1. Set up the environment  
    ```
    conda create -n nfe python=3.9
    conda activate nfe
    pip install -e .
    ```
2. Download data
    ```
    . scripts/download_all.sh
    ```
3. Experiments
    ```
    . scripts/run_all.sh
    ```
    or
    ```
    python -m nfe.train --experiment synthetic --data ellipse --model flow --flow-model [resnet/coupling/gru/mlp] --solver dopri5
    ```
    

### Results
| Model       | epoch_duration_mean | test_loss | loss_extrap_time | loss_extrap_space |
|-------------|----------------------|-----------|-------------------|-------------------|
| **ResNet**  | 0.02480             | 0.81185   | 1.48626          | 1.40874          |
| **Coupling**| 0.03418             | 1.04963   | 1.06937          | 1.20332          |
| **GRU**     | 0.02307             | 0.83711   | 0.82873          | 0.26828          |
| **MLP**     | 0.01526             | 0.83707   | 0.82859          | 0.26740          |


### Findings
ResNet Flow has the best test loss (0.81185). It is weak in temporal and spatial extrapolation. Coupling Flow is the slowest to learn (0.03418s/epoch). It has a large test loss. GRU Flow shows good performance overall. It has a good test loss (0.83711) and good extrapolation ability (loss_extrap_space: 0.26828). It has a balanced performance and time. MLP Flow is the fastest to learn (0.01526s/epoch). It has a competitive extrapolation performance similar to GRU Flow (loss_extrap_space: 0.26740).

ResNet Flow is the best choice for general test performance, but it struggles with extrapolation.
GRU Flow and MLP Flow provide strong extrapolation ability.
Coupling Flow has a slow learning speed.


## Problem 2
This is an implementation of a physics-informed neural network (PINN) for learning the Burger equation. It is based on the lecture note of IE552/AI603: Neural Differential Equations Physics Informed Deep Learning Dr. Sungil Kim.

### Implementation
```
python pinn.py
```

### Results
| Nu     |  Nf = 2000  |
|--------|-------------|
| 20     |  4.6e-01    |
| 40     |  3.6e-01    |
| 60     |  4.3e-01    |
| 80     |  3.3e-01    |
| 100    |  4.1e-01    |
| 200    |  3.4e-01    |

The reproduced results are similar to the original table. The differences may be due to hyperparameters.
