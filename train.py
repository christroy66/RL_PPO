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

gamma= 0.99
lambda_ = 0.95
clipping_value = 0.2
critic_discount = 0.5
entropy_beta = 0.001

def get_advantages(values, masks, rewards):
    returns = []
    gae = 0

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * masks[i] - values[i]
        gae = delta + gamma * lambda_ * masks[i] * gae
        returns.insert(0, gae+values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss (y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio*advantages
        p2 = K.clip(ratio, min_value=1-clipping_value, max_value=1+clipping_value) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))

        return total_loss
    
    return loss



def get_model_actor(input_dims, output_dims):
    #actor model to decide one action

    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims, ))
    advantages = Input(shape=(1, 1, ))
    rewards = Input(shape=(1, 1, ))
    values = Input(shape=(1, 1, ))


    x = Dense(10, activation="relu")(state_input)
    x = Dense(20, activation="relu")(x)
    x = Dense (10, activation="relu")(x)
    out_actions=Dense(n_actions, activation="softmax")(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values], outputs=out_actions)
    model.compile(optimizer=Adam(), loss=[ppo_loss(oldpolicy_probs=oldpolicy_probs, advantages=advantages, rewards=rewards, values=values)])
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
#print(state)
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



ppo_steps = 10 #time steps to train

states = []
actions = []
actions_probs = []
values = [] #value of critic model
masks = [] #indicates if game is over or not
rewards = []
actions_onehot = []

model_actor = get_model_actor(input_dims = state_dims, output_dims=n_actions)
model_critic = get_model_critic(input_dims = state_dims)

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))


for itr in range(ppo_steps):

    state_input = K.expand_dims(state, 0)
    action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps = 1)
    q_value = model_critic.predict([state_input], steps = 1)
    action = np.random.choice(n_actions, p=action_dist[0,:])
    action_onehot = np.zeros(n_actions)
    action_onehot[action] = 1
    #print(action)
    #print(action_dist)
    #print(state)


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
    actions_onehot.append(action_onehot)
    

    state = observation #change initial state after each iteration



    if done:
        env.reset()

state_input = K.expand_dims(state, 0)
q_value = model_critic.predict(state_input, steps=1)
values.append(q_value)

#calculate advantage from returns to train model actor
returns, advantages = get_advantages(values, masks, rewards)

model_actor.fit([states, actions_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
                [np.reshape(actions_onehot, newshape=(-1, n_actions))],
                verbose=True, shuffle=True, epochs=8)

model_critic.fit([states], [np.reshape(returns, newshape=(-1, 1))], verbose=True, shuffle=True, epochs=8)

#print("States", states)
#print("actions", actions)
#print("values", values)
#print("masks", masks)
#print("Reawards" , rewards)

env.close()

