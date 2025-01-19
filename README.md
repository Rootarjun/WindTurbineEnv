# Wind Turbine Optimization with PPO

## Project Overview

This project is a part of a collaborative academic paper on "Optimizing Renewable Energy Integration With Reinforcement Learning and Edge Computing". It demonstrates the use of **Proximal Policy Optimization (PPO)** to optimize wind turbine power generation by dynamically adjusting yaw and pitch angles. The environment simulates a simplified wind turbine system, and the PPO agent is trained to maximize power generation while minimizing unnecessary adjustments, which penalize the reward.

Wind turbines are a key component of renewable energy systems, and optimizing their operation can significantly enhance efficiency and power output. This project provides an approachable yet insightful way to explore reinforcement learning (RL) techniques for real-world energy applications.

---

## Features

1. **Custom Wind Turbine Environment**:  
   - Simulates wind speed, yaw, and pitch dynamics.  
   - Computes power generation using a simplified formula based on wind turbine physics.  
   - Provides a continuous state and action space for RL agents.  

2. **Proximal Policy Optimization (PPO) Agent**:  
   - Implements PPO for continuous action spaces.  
   - Includes an actor-critic architecture for policy and value estimation.  
   - Uses advantage normalization for stable training.  

3. **Performance Metrics**:  
   - Tracks total rewards, power generated, pitch adjustments, and yaw adjustments.  
   - Visualizes training results through various plots (e.g., rewards, losses, power output).  

4. **Model Saving**:  
   - Saves the trained actor and critic networks for reuse or further training.  

---

## Dependencies

- Python 3.7+
- [Gym](https://github.com/openai/gym)  
- [PyTorch](https://pytorch.org/)  
- NumPy  
- Matplotlib  

Install dependencies with:  
```bash
pip install -r requirements.txt
```

---

## How It Works

### Environment
The `WindTurbineEnv` simulates the dynamics of a wind turbine:
- **State Space**:  
  `[wind_speed, yaw_angle, pitch_angle]`  
  Wind speed ranges from 3 to 25 m/s, yaw from 0° to 360°, and pitch from -10° to 10°.
  
- **Action Space**:  
  `[yaw_adjustment, pitch_adjustment]`  
  Yaw adjustments range from -10° to 10°, and pitch adjustments range from -2° to 2°.

- **Reward Function**:  
  Power generation is calculated using a simplified physics model. The reward is the power generated minus a penalty for unnecessary yaw and pitch adjustments.

### PPO Agent
- The PPO agent uses an **actor-critic architecture** with feedforward neural networks.  
- The actor network outputs the action distribution, while the critic estimates state value.
- Training involves:
  - Calculating **advantages** based on returns and value estimates.
  - Clipping policy updates to prevent large deviations.
  - Minimizing the mean squared error for the value function.

### Training
- The agent interacts with the environment over multiple episodes (epochs).
- For each timestep, the agent selects actions, observes rewards, and stores trajectories.
- After each episode, the PPO agent updates its policy and value networks based on the collected trajectories.

---

## Results and Visualizations

1. **Rewards**:  
   Total rewards over training epochs are plotted to monitor learning progress.

2. **Losses**:  
   - Policy loss (actor network)  
   - Critic loss (advantage estimation)

3. **Power Generation**:  
   Plots showing power output over timesteps, normalized for visualization.

4. **Action Distributions**:  
   Histograms of pitch and yaw adjustments to analyze agent behavior.

---

## Usage

### Running the Project
1. Clone this repository.
2. Install the dependencies (maybe in a virtual environment).
3. Run the code:
   ```bash
   python wind_turbine_optimization.py
   ```
4. View training progress and plots.

---

## Future Work
- Integrate real-world wind turbine data for environment realism.
- Extend the reward function to include factors like blade wear or maintenance costs.
- Explore multi-agent setups for optimizing wind farms.

---

## License
This project is open-source and free to use for educational and research purposes.

--- 
Paper link - https://docs.google.com/document/d/1k6auOum3qsnu-vE1WTFi54E_e2MM7xW4/edit?usp=sharing&ouid=113082304330739098797&rtpof=true&sd=true
