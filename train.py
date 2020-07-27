import gym
import numpy as np
import keras.backend as K 
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam


#http://blog.varunajayasiri.com/ml/ppo_pytorch.html
#https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6
#https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6
#https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py


def get_model_actor(input_dims):

    state_input = Input(shape=input_dims)

    x = Dense(10, activation="relu")(state_input)
    x = Dense(20, activation="relu")(x)
    x = Dense (10, activation="relu")(x)

    out_actions=Dense(n_actions, activation="softmax")(x)

    model = Model(inputs=state_input, outputs=out_actions)
    model.compile(optimizer=Adam(), loss="mse")
    model.summary()

    return model


def get_model_critic(input_dims):

    state_input = Input(shape=input_dims)

    x = Dense(10, activation="relu")(state_input)
    x = Dense(20, activation="relu")(x)
    x = Dense (10, activation="relu")(x)

    out_actions=Dense(1, activation="tanh")(x)

    model = Model(inputs=state_input, outputs=out_actions)
    model.compile(optimizer=Adam(), loss="mse")
    model.summary()

    return model


env = gym.make("CartPole-v1")
state = env.reset()
state_dims = env.observation_space.shape
n_actions = env.action_space.n
#print(state_dims)
#  Observation:
    #     Type: Box(4)
    #     Num     Observation               Min                     Max
    #     0       Cart Position             -4.8                    4.8
    #     1       Cart Velocity             -Inf                    Inf
    #     2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    #     3       Pole Angular Velocity     -Inf                    Inf
# Actions:
    #     Type: Discrete(2)
    #     Num   Action
    #     0     Push cart to the left
    #     1     Push cart to the right



ppo_steps = 10

states = []
actions = []
actions_probs = []
values = [] #value of critic model
masks = [] #indicates if game is over or not
rewards = []

model_actor = get_model_actor(input_dims = state_dims)
model_critic = get_model_critic(input_dims = state_dims)


for itr in range(ppo_steps):

    state_input = K.expand_dims(state, 0)
    action_dist = model_actor.predict([state_input], steps = 1)
    q_value = model_critic.predict([state_input], steps = 1)
    action = np.random.choice(n_actions, p=action_dist[0,:])
    print(action)
    print(action_dist)


    env.render() #display and render the environment
    #action = env.action_space.sample() #sample a random action
    observation, reward, done, info = env.step(action) #return after each step
    mask = not done
    
    states.append(state)
    actions.append(action)
    values.append(q_value)
    masks.append(mask)
    rewards.append(reward)
    actions_probs.append(action_dist)

    state = observation #change initial state after each iteration

  

    if done:
        env.reset()

env.close()

