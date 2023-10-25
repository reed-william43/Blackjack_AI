import gym
import numpy as np
import pickle
import statistics
import matplotlib.pyplot as plt
import keras


env = gym.make("Blackjack-v1")
win_count = 0
lose_count = 0

#with open('model_pkl', 'rb') as f:
#    lr = pickle.load(f)
list = []
for i in range(100):
    for i in range(10):
        state = env.reset()
        episode = []
        while True:

            action = 0 if state[0] > 18 else 1
            print(state)
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                print("End game! Reward: ", reward)
                if reward > 0:
                    win_count = win_count+1
                    print("You won!\n")
                else:
                    lose_count = lose_count + 1
                    print("You lost\n")
                break
    list.append(win_count / (win_count + lose_count))   
            

#with open('model_pkl', 'wb') as files:
#    pickle.dump(env, files)

    perc = win_count / (win_count + lose_count)

    print("Win count: ", win_count)
    print("Lose count: ", lose_count)
    print("Win percentage: ", perc)    
   
#avg = statistics.mean(list)
#print(avg)

plt.plot(range(1,101), list)
plt.xlabel("Episode")
plt.ylabel("Win Percentage")
plt.title("Win Percentage Over Episodes")
plt.show()