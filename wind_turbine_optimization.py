import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os



# Environment: Define or use a simplified wind turbine environment
class WindTurbineEnv(gym.Env):
    def __init__(self):
        super(WindTurbineEnv, self).__init__()
        self.observation_space = gym.spaces.Box(
            low=np.array([3.0, 0.0, -10.0]),  # Wind speed, yaw, pitch (min values)
            high=np.array([25.0, 360.0, 10.0]),  # Wind speed, yaw, pitch (max values)
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-10.0, -2.0]),  # Adjustments for yaw and pitch (min values)
            high=np.array([10.0, 2.0]),  # Adjustments for yaw and pitch (max values)
            dtype=np.float32
        )
        self.state = None

    def reset(self):
        self.state = np.array([10.0, 180.0, 0.0])  # Initial wind speed, yaw, pitch
        return self.state

    def step(self, action):
        wind_speed, yaw, pitch = self.state
        yaw += action[0]
        pitch += action[1]
    
        #cliping the pitch and yaw within limits
        yaw=np.clip(yaw,0,360)
        pitch=np.clip(pitch,-10,10)

        # Calculate power coefficient (simplified example) and keeping it possitive
        cp = abs(math.cos(math.radians(pitch))*math.cos(math.radians(yaw)))
        power = cp * (wind_speed ** 3) * 0.5   # Simplified power production formula
        
        reward = power -(action[0]**2+action[1]**2)  #penalising for unnecessary movements

        self.state = np.array([wind_speed, yaw, pitch])
        done = False  # Continuous task; you can define terminal conditions if needed
        return self.state, reward, done, {}


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, model_dir,lr=3e-4, gamma=0.999, epsilon=0.2, policy_update_steps=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_update_steps = policy_update_steps
        self.model_dir=model_dir
        # Actor-Critic networks
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Softmax()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.policy_losses = []
        self.critic_losses = []

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mean = self.actor(state)
        dist = torch.distributions.Normal(mean, 0.1)  # Add a small standard deviation for exploration
        action = dist.sample()
        return action.detach().numpy(), dist.log_prob(action).sum().detach()

    def update(self, trajectories):
        states, actions, rewards, log_probs, dones = zip(*trajectories)
        returns=[0]*len(rewards)
        for i in range(len(rewards)-2,-1,-1):
            returns[i]+=returns[i+1]
            

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        returns = torch.tensor(returns,dtype=torch.float32)
        

        # Compute advantages
        values = self.critic(states).squeeze().detach()
        #advantages = rewards - values.detach()
        advantages = torch.zeros_like(returns, dtype=torch.float32, device=returns.device)
        lambd=0.95
        advantage = 0.0  # Initialize scalar

        for t in reversed(range(len(rewards))):
            td_error = returns[t] - values[t]  # TD error: difference between return and value
            advantage = td_error + self.gamma * lambd * advantage
            advantages[t] = advantage
        
        #Z- score normalisation for preventing abrupt policy changes
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.policy_update_steps):
            # Update policy (actor) batches updates
            new_log_probs = torch.distributions.Normal(self.actor(states), 0.1).log_prob(actions).sum(axis=-1)
            ratios = torch.exp(new_log_probs - log_probs)
            actor_loss = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            ).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            self.policy_losses.append(actor_loss.item())

            # Update value function (critic)
            critic_loss = nn.MSELoss()(self.critic(states).squeeze(), rewards)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
            self.critic_losses.append(critic_loss.item())
        
    def save_models(self,save_dir,model_name,model): 
        os.makedirs(save_dir, exist_ok=True)  # Creates the directory if it doesn't exist
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))

            
    def train(self, env, epochs=50, max_timesteps=100):
        reward_history = []
        power_generated = []
        pitch_adjustments = []
        yaw_adjustments = []
        
        #taking the radius of blades to be 50 m
        radius=50
        area=np.pi*(radius**2)
        #Initialising the power coefficient and air density(row)
        Cp=0.59
        row=1.225
        #Initialising the total time taken per epoch list
        times=[]

        for epoch in range(epochs):
            start_time=time.time()
            state = env.reset()
            trajectories = []
            total_reward = 0

            for _ in range(max_timesteps):
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                trajectories.append((state, action, reward, log_prob, done))
                state = next_state
                total_reward += reward

                # Record metrics
                updated_area=area*math.cos(math.radians(state[1]))*math.cos(math.radians(state[2]))  #Calculating the updated area
                power=abs(0.5*updated_area*Cp*row*(state[0]**3)) #calculating the power generated
                power_generated.append(power)
                pitch_adjustments.append(action[1])
                yaw_adjustments.append(action[0])

                if done:
                    break

            reward_history.append(total_reward)
            self.update(trajectories)
            time_taken=time.time()-start_time
            times.append(time_taken)
            print(f"Epoch {epoch + 1}/{epochs}, Total Reward: {total_reward}")

        # Save models
        
        self.save_models(self.model_dir,"wind_actor.pth",self.actor)
        self.save_models(self.model_dir,"wind_critic.pth",self.critic)
        

        print("Average time per epoch : ",sum(times)/len(times))

        # Plot results
        self.plot_results(reward_history, power_generated, pitch_adjustments, yaw_adjustments)
        


    def plot_results(self, reward_history, power_generated, pitch_adjustments, yaw_adjustments):
        

        # Min-max normalization for policy loss
        policy_losses_min = min(self.policy_losses)
        policy_losses_max = max(self.policy_losses)
        self.policy_losses= [(x - policy_losses_min) / (policy_losses_max - policy_losses_min + 1e-8) for x in self.policy_losses]

        # Min-max normalization for critic loss
        critic_losses_min = min(self.critic_losses)
        critic_losses_max = max(self.critic_losses)
        self.critic_losses= [(x - critic_losses_min) / (critic_losses_max - critic_losses_min + 1e-8) for x in self.critic_losses]
        # Plot cumulative reward
      
        plt.plot(reward_history)
        plt.title("Total Reward per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Total Reward")
        plt.show()
        
        # Plot policy loss
        plt.plot(self.policy_losses[0:50], label="Policy Loss", color="blue")
        plt.title("Policy Loss Over Updates -- Actor Network")
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Plot critic loss
        plt.plot(self.critic_losses[0:501:10], label="Advantage Loss (Critic)", color="green")
        plt.title("Advantage Loss Over Updates -- Critic Network")
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Plot power generated and limiting the data points
  
        plt.plot(power_generated[0:50000:500],color='blue',  linestyle='-')
        plt.title("Power Generated Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Power Generated")
        plt.show()
       
        
        # Plot pitch adjustments
     
        plt.hist(pitch_adjustments,bins=10)
        plt.title("Pitch Adjustments Over TIme")
        plt.xlabel("Pitch Adjustment Ranges")
        plt.ylabel("Frequency")
        plt.show()
        # Plot yaw adjustments
       
        plt.hist(yaw_adjustments,bins=10)
        plt.title("Yaw Adjustments Over Time")
        plt.xlabel("Yaw Adjustment Ranges")
        plt.ylabel("Frequency")

       
        plt.show()
        print("average return",sum(reward_history)/len(reward_history))
        print("Max power generated",max(power_generated))
        print("Average pitch adjustments",sum(pitch_adjustments)/len(pitch_adjustments))
        print("Average yaw adjustments",sum(yaw_adjustments)/len(yaw_adjustments))
        
        total_params_actor = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)

        total_params_critic = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        print("total params",total_params_actor,total_params_critic)
        
        print("\nFinal Policy Parameters (Actor Network):")
        for name, param in self.actor.named_parameters():
            print(f"{name}: {param.data}")
       
       
        

# Main
def main():
    env = WindTurbineEnv()
    agent = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],model_dir="Models_Wind")
    agent.train(env, epochs=50, max_timesteps=500)


if __name__ == "__main__":
    main()
