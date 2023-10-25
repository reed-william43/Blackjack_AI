import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import gymnasium as gym
from collections import defaultdict
import random
import enum


# Set up environment
env = gym.make("Blackjack-v1", sab=True)
done = False
observation, info = env.reset()

action = env.action_space.sample()
observation, reward, terminated, truncacted, info = env.step(action)

# Create a Q-learning agent - class taken from {source}, parameters changed for better training
class BlackjackAgent:
    def __init__(
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
        
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # card counting logic
        
        
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
        self,
        obs: tuple[int, int, bool],
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
        
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


ranks = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "jack": 10,
    "queen": 10,
    "king": 10,
    "ace": (1,11)
}

class Suits(enum.Enum):
    spades = "spades"
    clubs = "clubs"
    hearts = "hearts"
    diamonds = "diamonds"

class Cards:
    def __init__(self, rank, suit, value):
        self.suit = suit,
        self.rank = rank,
        self.value = value
        
    def __str__(self):
        return self.rank + "of " + self.suit.value
    
class Deck:
    def __init__(self, num=2):
        self.cards=[]
        for i in range(num):
            for suit in Suits:
                for rank, value in ranks.items():
                    self.cards.append(Cards(suit, rank, value))
                    
    def shuffle_deck(self):
        random.shuffle(self.cards)
        
    def deal_cards(self):
        return self.cards.pop(0)
    
    def peek(self):
        if len(self.cards) > 0:
            return self.cards[0]
    
    

# Define plotting functions
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
    


# Set learning rate and amount(number of games)
learning_rate = 0.0001
num_episodes = 1000000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (num_episodes / 2)
final_epsilon = 0.1

agent = BlackjackAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncacted, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncacted
        obs = next_obs
    
    agent.decay_epsilon
    
value_grid, policy_grid = make_grids(agent, usable_ace=True)
fig1 = make_plots(value_grid, policy_grid, title="With usable ace")
plt.show()

value_grid, policy_grid = make_grids(agent, usable_ace=True)
fig2 = make_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()