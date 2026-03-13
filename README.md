# Mario Double DQN agent
A Reinforcement Learning agent for Super Mario Bros using Double Deep Q-Networks (DDQN).

## Usage
### 1. Training
Train the agent with default hyperparameters:
```bash
python train.py
```
To train using optimized hyperparameters from Optuna:
```bash
python train.py --optimized
```

### 2. Hyperparameter Optimization
Search for the best hyperparameters using Optuna:
```bash
python optimize.py
```

### 3. Evaluation
Compare the performance of default and optimized agents:
```bash
python evaluate.py
```

## Repository Structure
- `train.py`: Main script to train the model.
- `optimize.py`: Script for hyperparameter optimization.
- `evaluate.py`: Script to evaluate and compare models with statistical tests.
- `config.yaml`: Configuration for environment, training, and optimization.
- `src/`: Contains agent implementation, wrappers, and configuration logic.
