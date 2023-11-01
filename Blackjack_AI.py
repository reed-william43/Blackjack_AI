import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import gymnasium as gym
from collections import defaultdict
import random

# Set up environment
env = gym.make("Blackjack-v1", sab=True)
done = False
observation, info = env.reset()

player_hand = []

# Determine which cards player has in hand (for card counting)
for _ in range(2): # range of 2 because of number of cards in hand
    player_hand.append(observation[0]) # creates a list of the card values in the players hand - extracted from the tuple returned by the action in the gym environment
    print(player_hand)
    action = env.action_space.sample()
    observation, reward, terminated, truncacted, info = env.step(action) 
    
# Create a Q-learning agent - class taken from [1], parameters changed for better training
class BlackjackAgent:
    def __init__( #[1]
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.lr = learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.training_error = []
        self.card_count = 0.0
        self.player_hand = []
        self.dealer_card = 0
        
    def get_action(self, obs: tuple[int, int, bool], player_hand, dealer_card) -> int:
        # card counting logic
        print("get_action")
        self.card_count += self.dealer_check(dealer_card)
        self.card_count += self.player_check(player_hand)
        print("p hand: ", player_hand)
       
        min_cc = -10
        max_cc = 10
        if self.card_count > max_cc:
            self.card_count += 1.0
        elif self.card_count < min_cc:
            self.card_count -= 1.0
        else:
            self.card_count = (self.card_count - min_cc) / (max_cc - min_cc)

        if self.card_count > 0:
            return int(np.argmax(self.q_values[obs]))
        elif self.card_count < 0:
            return int(np.argmin(self.q_values[obs]))
        else:
            return env.action_space.sample()

    def update_cc(self, card):
        for cards in card:
            if cards >= 2 and cards <= 6:
                self.card_count += 1
            elif cards >= 7 and cards <= 9:
                self.card_count += 0
            else:
                self.card_count -= 1
        

    
    def dealer_check(self, dealer_card):
        num_ace = 0
        use_one = 0
        
        if dealer_card >= 2 and dealer_card <= 6:  # if dealer has a card in the range 2-6
            self.card_count += 1
        elif dealer_card >=7 and dealer_card <= 9: # if dealer has a card in the range 7-9
            self.card_count += 0
        else:                                      # if dealer has a card in the rand 10-A
            self.card_count -= 1
            
        print("cc: ", self.card_count)
        return self.card_count
                        
    def player_check(self, player_hand):
        num_ace = 0
        use_one = 0
        
        for card in player_hand:
            if card >= 2 and card <= 6:  # if dealer has a card in the range 2-6
                self.card_count += 1
            elif card >=7 and card <= 9: # if dealer has a card in the range 7-9
                self.card_count += 0
            else:                        # if dealer has a card in the rand 10-A
                self.card_count -= 1       
        return self.card_count
        
    def update( #[1]
        self,
        obs: tuple[int, int, bool], # [player sum, dealer showing, usable ace(T/F)]
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )
        
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
        
    def decay_epsilon(self): #[1]
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# Define plotting functions [1]
def make_grids(agent, usable_ace=False):
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))
        
    player_count, dealer_count = np.meshgrid(
        np.arange(12, 22),
        np.arange(1, 11)
    )
    
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr = np.dstack([player_count, dealer_count])
    )
    value_grid = player_count, dealer_count, value
    
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count])
    )
    return value_grid, policy_grid

def make_plots(value_grid, policy_grid, title: str):
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)
    
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap = "viridis",
        edgecolor = "none"
    )
    plt.xticks(range(12,22), range(12,22))
    plt.yticks(range(1,11), ["A"] + list(range(2,11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=99)
    ax1.view_init(20, 220)
    
    fig.add_subplot(1,2,2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12,22))
    ax2.set_yticklabels(["A"] + list(range(2,11)), fontsize=12)
    
    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick")
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3,1))
    return fig
    
# Set learning rate and amount(number of games) [1]
learning_rate = 0.0001
num_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (num_episodes / 2)
final_epsilon = 0.1

agent = BlackjackAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


card_count = 0
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        player_hand = []
        player_hand.append(obs[0])
        action = agent.get_action(obs, player_hand, obs[1])
        next_obs, reward, terminated, truncacted, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncacted
        obs = next_obs
        print(player_hand)

    
    agent.decay_epsilon

# Create plots for "with usable ace" [1]
value_grid_with_ace, policy_grid_with_ace = make_grids(agent, usable_ace=True)
fig1 = make_plots(value_grid_with_ace, policy_grid_with_ace, title="With usable ace")
plt.show()

# Create plots for "without usable ace" [1]
value_grid_without_ace, policy_grid_without_ace = make_grids(agent, usable_ace=False)
fig2 = make_plots(value_grid_without_ace, policy_grid_without_ace, title="Without usable ace")
plt.show()


'''
Resources:
    [1] https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
'''