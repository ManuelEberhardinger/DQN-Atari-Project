import gym
import math
import random
import numpy as np
from itertools import count
from DQN import DQN, DuelingDQN, ReplayMemory, Transition
from helper import ImageProcessor, video_callable
import torch
import torch.optim as optim
from torch.autograd import Variable
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions for Breakout
env = gym.make("Pong-v0")


#start video recording
video_path = os.getcwd() + "/recording-ddqn-dueling-pong"
env = gym.wrappers.Monitor(env, video_path, video_callable=video_callable)

env.reset()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Started on device:", device)

# 84 is the used size in DQN paper
image_processor = ImageProcessor(84, device)

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 100000
TARGET_UPDATE = 1000

IN_CHANNELS = 4

policy_net = DuelingDQN(IN_CHANNELS, env.action_space.n).float().to(device)
target_net = DuelingDQN(IN_CHANNELS, env.action_space.n).float().to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
memory = ReplayMemory(100000)

LEARNING_STARTS = 10000

steps_done = 0
steps_trained = 0


def select_action(current_state):
    global steps_done
    global steps_trained
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_trained / EPS_DECAY)
    print(eps_threshold)
    if LEARNING_STARTS < steps_done:
        steps_trained += 1

    steps_done += 1
    if LEARNING_STARTS < steps_done and sample > eps_threshold:
        with torch.no_grad():
            return policy_net(current_state).max(1)[1].view(-1, 1)
    else:
        return torch.LongTensor([[random.randrange(env.action_space.n)]]).to(device)


def optimize_model(double_dqn=False):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch
    batch = Transition(*zip(*transitions))
    state_batch = Variable(torch.from_numpy(np.asarray(batch.state)).to(device))
    action_batch = Variable(torch.from_numpy(np.asarray(batch.action)).long().to(device))
    reward_batch = Variable(torch.from_numpy(np.asarray(batch.reward)).to(device))
    done_batch = Variable(torch.from_numpy(np.asarray(batch.done)).float().to(device))
    next_state_batch = Variable(torch.from_numpy(np.asarray(batch.next_state)).to(device))

    q_s_a = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    q_s_a = q_s_a.squeeze()

    if double_dqn:
        # get the Q values for best actions in next_state_batch
        # based off the current Q network
        # max(Q(s', a', theta_i)) wrt a'
        q_next_state_values = policy_net(next_state_batch).detach()
        _, a_prime = q_next_state_values.max(1)
        # get Q values from frozen network for next state and chosen action
        # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
        q_target_next_state_values = target_net(next_state_batch).detach()
        q_target_s_a_prime = q_target_next_state_values.gather(1, a_prime.unsqueeze(1))
        q_target_s_a_prime = q_target_s_a_prime.squeeze()

        # if current state is end of episode, then there is no next Q value
        q_target_s_a_prime = (1 - done_batch) * q_target_s_a_prime

        error = (reward_batch + (GAMMA * q_target_s_a_prime)) - q_s_a
    else:
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        next_state_values = target_net(next_state_batch).detach()
        q_s_a_prime, a_prime = next_state_values.max(1)
        q_s_a_prime = (1 - done_batch) * q_s_a_prime

        # Compute Bellman error
        # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
        error = (reward_batch + (GAMMA * q_s_a_prime)) - q_s_a

    # clip the error and flip
    clipped_error = -1.0 * error.clamp(-1, 1)
    # Optimize the model
    optimizer.zero_grad()

    q_s_a.backward(clipped_error.data)

    optimizer.step()


num_episodes = 10000
for i_episode in range(num_episodes):
    print("Episode: ", i_episode)
    # Initialize the environment and state
    state = env.reset()
    state = image_processor.process(state)
    state = np.stack([state]*4, axis=0)
    for t in count():
        # Select and perform an action
        action = select_action(torch.FloatTensor(state).unsqueeze(0).to(device))
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.Tensor([reward], device=device)
        print("Reward:", reward.item(), "Episode:", i_episode, "Steps done:", steps_done)
        # Observe new state
        next_state = image_processor.process(next_state)
        next_state = np.append(state[1:, :, :], np.expand_dims(next_state, 0), axis=0)

        if done is False:
            done = 0
        else:
            done = 1

        # Store the transition in memory
        memory.push(state, action, next_state, reward, done)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()

        # Update the target network
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            print("Done")
            break

    torch.save(target_net.state_dict(), os.getcwd() + '/pong_dueling_dqn')


torch.save(target_net.state_dict(), os.getcwd() + '/pong_dueling_dqn')
print('Complete')
env.close()
