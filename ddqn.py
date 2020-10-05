import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from utility import ElecTradingEnv as Env
import tensorflow as tf

EPISODES = 1000
ROUTE = './data/DDQN_trading/model/'


class DDQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

dependence_A = []
dependence_B = []
dependence_C = []
dependence_D = []
sustain_A = []
sustain_B = []
sustain_C = []
sustain_D = []
battery_A = []
battery_B = []
battery_C = []
battery_D = []


def append_dependence(dependence):
    dependence_A.append(dependence[0])
    dependence_B.append(dependence[1])
    dependence_C.append(dependence[2])
    dependence_D.append(dependence[3])


def append_sustain(sustain):
    sustain_A.append(sustain[0])
    sustain_B.append(sustain[1])
    sustain_C.append(sustain[2])
    sustain_D.append(sustain[3])


def append_battery(battery):
    battery_A.append(battery[0])
    battery_B.append(battery[1])
    battery_C.append(battery[2])
    battery_D.append(battery[3])


if __name__ == "__main__":
    env = Env()
    state_size = env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size)
    total_score = np.array([])
    mean = []
    count = []
    #env.render_count_energy()
    #env.render_battery_cycle()
    #env.render_sustain()
    #env.render_dependence()
    # agent.load("Trading_ddqn.h5")

    done = False
    batch_size = 64
    for e in range(EPISODES):
        state = env.reset()
        score = 0
        k = 1
        for time in range(2000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                """append measurements"""
                mean.append(env.return_mean_score())
                dependence = env.return_dependence_factor()
                count_energy = env.return_count_energy()
                sustain = env.return_sustain_factor()
                battery = env.return_battery_cycle_factor()

                count.append(count_energy)

                append_dependence(dependence)
                append_sustain(sustain)
                append_battery(battery)

                """save ESS & price"""
                env.save_ESS_week()
                env.save_price_week()

                """update and print"""
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, score, agent.epsilon))
                break

            if e == (2 - 1):
                env.save_episodes_count_energy(count)
                env.save_sustain_factor(sustain_A,sustain_B,sustain_C,sustain_D)
                env.save_dependence_factor(dependence_A,dependence_B, dependence_C, dependence_D)
                env.save_battery_life(battery_A, battery_B, battery_C, battery_D)
                env.save_mean_score(mean)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            #if e % 10 == 0:
                #agent.save(ROUTE + "Trading_ddqn.h5")