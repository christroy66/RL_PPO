import gym

#http://blog.varunajayasiri.com/ml/ppo_pytorch.html
#https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6
#https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6
#https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py


env = gym.make("CartPole-v1")

state_dims = env.observation_space.high#.shape

#Observations Box(4)
    #Cart Postion
    #1 Cart Velocity
    #2 Pole Angle
    #3 Pole Angular velocity
n_actions = env.action_space#.n

print(state_dims)
print(n_actions)

state = env.reset()





for _ in range(100):

    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    #print(observation)
    #print(info)

    if done:
        env.reset()

env.close()

