import gym
import pandas as pd
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
import pygame

# Agent Num & plot x axis max
Agent_N = 4
x_max = 52
ROUTE_MEASURE = './data/DDQN_trading/measurement/'
ROUTE_SAVE = './data/DDQN_trading/save_ESS_price/'
ROUTE_SCORE = './data/DDQN_trading/score/'
ROUTE_MODEL = './data/DDQN_trading/model/'


def calculate_standard_price():
    """calculate original price of MGs"""
    price = []
    original_loan = 150000000
    MG_1_ESS_price = original_loan * 6
    MG_2_ESS_price = original_loan * 6
    MG_3_ESS_price = original_loan * 6
    MG_4_ESS_price = original_loan * 6

    MG_1_price_year = MG_1_ESS_price/25
    MG_2_price_year = MG_2_ESS_price/25
    MG_3_price_year = MG_3_ESS_price/25
    MG_4_price_year = MG_4_ESS_price/25

    MG_1_price_day = MG_1_price_year/365
    MG_2_price_day = MG_2_price_year/365
    MG_3_price_day = MG_3_price_year/365
    MG_4_price_day = MG_4_price_year/365

    price.append(int(MG_1_price_day/2000))
    price.append(int(MG_2_price_day/2000))
    price.append(int(MG_3_price_day/2000))
    price.append(int(MG_4_price_day/2000))

    return price


# original prices array of MGs
standard_price = calculate_standard_price()

# position of the game(x axis, y axis)
x = 512
y = 320

# max step from environment
MAX_STEP = 365

# notation of ESS max (M/W)
W_SmartA_max = 10
W_SmartB_max = 8
W_SmartC_max = 10
W_SmartD_max = 8
W = [W_SmartA_max, W_SmartB_max, W_SmartC_max, W_SmartD_max]
W_max = np.array(W)

# transmission loss among MGs
distance = [[0, 1, 0.99, 1],
            [1, 0, 0.98, 0.99],
            [0.99, 0.98, 0, 1],
            [1, 0.99, 1, 0]]

MG_COST_MAP = np.array(distance)

# state size & action size
state_size = Agent_N * 5
action_size = 10


class ElecTradingEnv(gym.Env):
    # action 공간 설정, state 공간 설정
    action_space = spaces.Discrete(action_size)
    observation_space = np.zeros([1, state_size])

    def __init__(self):
        # Parameter 선언
        self.count_energy_step = None
        self.count_energy = None
        self.win = None
        self.save_step = None
        self.save_reward = None
        self.save_action = None
        self.save_ESS_1 = None
        self.save_ESS_2 = None
        self.save_ESS_3 = None
        self.save_ESS_4 = None
        self.load_step = []
        self.load_reward = None
        self.load_action = None
        self.load_ESS_1 = None
        self.load_ESS_2 = None
        self.load_ESS_3 = None
        self.load_ESS_4 = None
        self.clock = pygame.time.Clock()
        self.current_step = None
        self.action = None
        self.done = False
        self.ESS = None
        self.price_ratio = np.random.rand(Agent_N, 1)
        self.ESS_reward = None
        self.reward = None
        self.obs = None
        self.agent = None
        self.total_reward = None
        self.dependence_count_A = None
        self.dependence_count_B = None
        self.dependence_count_C = None
        self.dependence_count_D = None
        self.total_dependence = None
        self.sustain_count_A = None
        self.sustain_count_B = None
        self.sustain_count_C = None
        self.sustain_count_D = None
        self.total_sustain = None
        self.count = 0
        self.MG_A = []
        self.MG_B = []
        self.MG_C = []
        self.MG_D = []
        self.MG_A_week = []
        self.MG_B_week = []
        self.MG_C_week = []
        self.MG_D_week = []
        self.score_plt = []
        self.reward_mean = np.array([])
        self.total_mean = 0
        self.score_plt_week = []
        self.price_plt_MG_A = []
        self.price_plt_MG_B = []
        self.price_plt_MG_C = []
        self.price_plt_MG_D = []
        self.score_plt_ep = []
        self.step_ = np.linspace(1, 1051, 1051)

        # factor = 0.28
        self.trade_price_factor = (self.cal_kelly(0.6) + 1 + 1)/10

        self.prev_states = np.random.rand(Agent_N, 5)
        self.price = np.random.rand(Agent_N, 1)
        self.cur_states = np.random.rand(Agent_N, 5)
        self.df = pd.read_excel('1day_refer.xlsx', sheet_name='Sheet3', header=1)

    def _next_observation(self):
        """take action of next observation for next states"""

        self.cur_states = self.get_states_agent(self.df, self.price)
        self.obs = np.reshape(self.cur_states, (1, state_size))
        return self.obs

    def gaussian_calculation(self, ESS):
        """return 0.1 ~ 0.398 factor of reward when ESS is in range 20% ~ 80%"""
        reward = (1 / np.sqrt(2 * np.pi)) * np.exp(- (ESS - 0.55) ** 2 / 0.065)  # 평균은 0.55로 설
        return reward

    def sustain_count(self, MG):
        """check MG 35 ~ 85%"""
        if MG == 0:
            if 0.25 < self.agent[MG][0] < 0.85:
                self.sustain_count_A += 1
                self.total_sustain += 1
        if MG == 1:
            if 0.25 < self.agent[MG][0] < 0.85:
                self.sustain_count_B += 1
                self.total_sustain += 1
        if MG == 2:
            if 0.25 < self.agent[MG][0] < 0.85:
                self.sustain_count_C += 1
                self.total_sustain += 1
        if MG == 3:
            if 0.25 < self.agent[MG][0] < 0.85:
                self.sustain_count_D += 1
                self.total_sustain += 1

    def ESS_calculation(self):
        """calculate ess_reward through gaussian function, maximum reward summation is 1.598(4 Agents)"""
        agent_ess_reward = 0
        for i in range(0, Agent_N):
            ESS = self.agent[i][0]
            agent_ess_reward += self.gaussian_calculation(ESS)
            self.sustain_count(i)

        return agent_ess_reward

    def trading_calculation_one_seller(self, MG_one, MG_another, trading_energy_one):
        """when MG_one is seller, calculate trade energy and calculate price"""
        trading_energy_one_real = W_max[MG_one] * trading_energy_one
        check = trading_energy_one_real/W_max[MG_another] + self.agent[MG_another][0]

        if check > 1:
            overflow = (check - 1) * W_max[MG_another]
            trade_real_energy = trading_energy_one_real - overflow
            trade_energy = trade_real_energy / W_max[MG_one]  # 실제 거래 가능 에너지 %(MG_one 기준)

        else:
            trade_energy = trading_energy_one_real / W_max[MG_one]
            trade_real_energy = trading_energy_one_real  # 에너지를 제공했기 때문에 0이 됨

        energy_one = -trade_real_energy / W_max[MG_one]
        energy_another = trade_real_energy / W_max[MG_another]

        self.action_one_seller_profit_price_calculation(MG_one, MG_another, trade_energy)

        trade_energy *= W_max[MG_one]
        self.count_trading_energy(trade_energy)

        return trade_energy, energy_one, energy_another

    def trading_calculation_another_seller(self, MG_one, MG_another, trading_energy_another):
        """when MG_another is seller, calculate trade energy and calculate price"""
        trading_energy_another_real = W_max[MG_another] * trading_energy_another
        check = trading_energy_another_real / W_max[MG_one] + self.agent[MG_one][0]

        if check > 1:
            overflow = (check - 1) * W_max[MG_one]
            trade_real_energy = trading_energy_another_real - overflow
            trade_energy = trade_real_energy/W_max[MG_another]

        else:
            trade_energy = trading_energy_another_real/W_max[MG_another]
            trade_real_energy = trading_energy_another_real  # 에너지를 제공했기 때문에 0이 됨

        energy_one = trade_real_energy / W_max[MG_one]
        energy_another = -trade_real_energy / W_max[MG_another]

        self.action_another_seller_profit_price_calculation(MG_one, MG_another, trade_energy)

        trade_energy *= W_max[MG_another]
        self.count_trading_energy(trade_energy)

        return trade_energy, energy_one, energy_another

    def action_one_seller_profit_price_calculation(self, MG_one, MG_another, trade_energy):
        """when MG_one is seller, calculate price"""
        MG_one_energy = self.agent[MG_one][0] - trade_energy

        A = 2 - self.agent[MG_one][4]
        self.price_ratio[MG_one] = A * (self.trade_price_factor / (1 + MG_one_energy)) - 0.04

        trade_energy_real = trade_energy * W_max[MG_one]
        trade_energy = trade_energy_real / W_max[MG_another]
        MG_another_energy = trade_energy + self.agent[MG_another][0]

        B = 2 - self.agent[MG_another][4]
        self.price_ratio[MG_another] = B * (self.trade_price_factor / (1 + MG_another_energy)) - 0.04

    def action_another_seller_profit_price_calculation(self, MG_one, MG_another, trade_energy):
        """when MG_another is seller, calculate price"""
        MG_another_energy = self.agent[MG_another][0] - trade_energy

        B = 2 - self.agent[MG_another][4]
        self.price_ratio[MG_another] = B * (self.trade_price_factor / (1 + MG_another_energy)) - 0.04

        trade_energy_real = trade_energy * W_max[MG_another]
        trade_energy = trade_energy_real/W_max[MG_one]
        MG_one_energy = self.agent[MG_one][0] + trade_energy

        A = 2 - self.agent[MG_one][4]
        self.price_ratio[MG_one] = A * (self.trade_price_factor / (1 + MG_one_energy)) - 0.04

    def action_profit_calculation(self, MG_one, MG_another):
        """calculate profit through comparison of trading energy between MG_one and MG_another"""
        trading_energy_one = self.agent[MG_one][1]  # i 의 trading 량
        trading_energy_one_real = W_max[MG_one] * trading_energy_one
        trading_energy_another = self.agent[MG_another][1]  # j 의 trading 량
        trading_energy_another_real = W_max[MG_another] * trading_energy_another

        summation = []
        summation.append(self.agent[MG_one][2])
        summation.append(self.agent[MG_another][2])
        trade_price = np.mean(summation)  # 평균 가격 사용

        if trading_energy_one_real > trading_energy_another_real:  # MG_one is seller, MG_another is buyer
            trade_energy, energy_one, energy_another = \
                self.trading_calculation_one_seller(MG_one, MG_another, trading_energy_one)

        elif trading_energy_one_real < trading_energy_another_real:  # MG_one is buyer, MG_another is seller
            trade_energy, energy_one, energy_another = \
                self.trading_calculation_another_seller(MG_one, MG_another, trading_energy_another)

        else:  # trading_energy_i = trading_energy_j  같은 경우
            if self.agent[MG_one][0] > self.agent[MG_another][0]:  # i가 갖고 있는 ESS 가 많은 쪽이 에너지를 공급
                trade_energy, energy_one, energy_another = \
                    self.trading_calculation_one_seller(MG_one, MG_another, trading_energy_one)

            elif self.agent[MG_one][0] < self.agent[MG_another][0]:  # j가 갖고 있는 ESS 가 많은 쪽이 에너지를 공급
                trade_energy, energy_one, energy_another = \
                    self.trading_calculation_another_seller(MG_one, MG_another, trading_energy_another)

            else:
                trade_energy = 0  # Energy 거래 x , profit 계산시 사용
                A = 2 - self.agent[MG_one][4]
                self.price_ratio[MG_one] = A * self.trade_price_factor / (1 + (self.agent[MG_one][0])) - 0.04
                B = 2 - self.agent[MG_another][4]
                self.price_ratio[MG_another] = B * self.trade_price_factor / (1 + (self.agent[MG_another][0])) - 0.04
                energy_one = 0
                energy_another = 0

        self.agent[MG_one][0] = np.round(self.agent[MG_one][0] + energy_one, 2)
        self.agent[MG_another][0] = np.round(self.agent[MG_another][0] + energy_another, 2)

        profit = (trade_price / np.min(summation)) * np.log(trade_energy * distance[MG_one][MG_another] + 1)
        profit = np.round(profit, 2)

        return profit

    def profit_price_calculation(self, MG_one):
        """price calculating of MGs which do not attend trading"""
        A = 2 - self.agent[MG_one][4]
        self.price_ratio[MG_one] = A * self.trade_price_factor / (1 + self.agent[MG_one][0]) - 0.04

    # 현재 state, action 에서의 reward 계산
    def profit_calculation(self, action):
        """get action and calculate profit of actions"""
        if action == 0:  # no trading
            profit = 0
            for i in range(Agent_N):
                self.profit_price_calculation(i)

        elif action == 1:  # ab
            MG_one = 0
            MG_another = 1
            profit = self.action_profit_calculation(MG_one, MG_another)
            self.profit_price_calculation(2)
            self.profit_price_calculation(3)

        elif action == 2:  # ac
            MG_one = 0
            MG_another = 2
            profit = self.action_profit_calculation(MG_one, MG_another)
            self.profit_price_calculation(1)
            self.profit_price_calculation(3)

        elif action == 3:  # ad
            MG_one = 0
            MG_another = 3
            profit = self.action_profit_calculation(MG_one, MG_another)
            self.profit_price_calculation(1)
            self.profit_price_calculation(2)

        elif action == 4:  # bc
            MG_one = 1
            MG_another = 2
            profit = self.action_profit_calculation(MG_one, MG_another)
            self.profit_price_calculation(0)
            self.profit_price_calculation(3)

        elif action == 5:  # bd
            MG_one = 1
            MG_another = 3
            profit = self.action_profit_calculation(MG_one, MG_another)
            self.profit_price_calculation(0)
            self.profit_price_calculation(2)

        elif action == 6:  # cd
            MG_one = 2
            MG_another = 3
            profit = self.action_profit_calculation(MG_one, MG_another)
            self.profit_price_calculation(0)
            self.profit_price_calculation(1)

        elif action == 7:  # ab&cd
            MG_one = 0
            MG_another = 1
            profit = self.action_profit_calculation(MG_one, MG_another)
            MG_one = 2
            MG_another = 3
            profit += self.action_profit_calculation(MG_one, MG_another)

        elif action == 8:  # ac&bd
            MG_one = 0
            MG_another = 2
            profit = self.action_profit_calculation(MG_one, MG_another)
            MG_one = 1
            MG_another = 3
            profit += self.action_profit_calculation(MG_one, MG_another)

        else:  # ad&bc
            MG_one = 0
            MG_another = 3
            profit = self.action_profit_calculation(MG_one, MG_another)
            MG_one = 1
            MG_another = 2
            profit += self.action_profit_calculation(MG_one, MG_another)

        return profit

    # 현재 action 에 따라 가격을 변화시킨다.
    def price_calculation(self):
        """change price of MGs with price ratio that changed before"""
        for i in range(Agent_N):
            self.price[i] = standard_price[i] * (1 + self.price_ratio[i])

    def MG_save(self):
        """MG save for poltting"""
        for i in range(Agent_N):
            self.ESS[i] = self.agent[i][0]

        self.MG_A.append(int(100 * self.ESS[0]))
        self.MG_B.append(int(100 * self.ESS[1]))
        self.MG_C.append(int(100 * self.ESS[2]))
        self.MG_D.append(int(100 * self.ESS[3]))

    def price_save(self):
        """price save for poltting"""
        self.price_plt_MG_A.append(self.cur_states[0][2])
        self.price_plt_MG_B.append(self.cur_states[1][2])
        self.price_plt_MG_C.append(self.cur_states[2][2])
        self.price_plt_MG_D.append(self.cur_states[3][2])

    # agents 로부터 받은 action 을 step 에 넣고 reward 를 받는다.
    def step(self, action):
        """get action from ddqn agent and return next obs, reward, done"""
        self.action = action
        done = False
        self.current_step += 1

        profit_reward = self.profit_calculation(action)  # 현재 action 에 따른 reward 계산
        self.ESS_reward = np.round(self.ESS_calculation(), 2) # ESS 잔량의 reward 반환

        alpha = 0.3
        profit_reward = alpha * profit_reward + (1 - alpha) * self.ESS_reward

        self.reward = profit_reward  # 위에 둘의 비율로 최대 1의 reward 를 받는다.
        self.price_calculation()  # MG의 가격 변동

        self.reward_mean = np.append(self.reward_mean, self.reward)

        self.MG_save()
        self.price_save()

        self.total_reward += self.reward
        self.score_plt.append(self.total_reward)
        next_state = self._next_observation()

        # MAX_STEP 보다 클 시, episode 를 종료
        if self.current_step > MAX_STEP:
            self.save_measurements()
            done = True

        return next_state, self.reward, done, {}

    def cal_kelly(self, prob_win):
        """kelly calculation"""
        betting = 2  # 고정된 손익비로 생각
        double_Kelly_factor = 2 * (prob_win - (1 - prob_win) / betting)
        return double_Kelly_factor

    def cal_prob_win(self, weather):
        """calculate probability from weather"""
        if weather == 1:  # double_Kelly_factor = 0.8
            prob_win = 0.6
        elif weather == 0.8:  # double_Kelly_factor = 0.65
            prob_win = 0.55
        elif weather == 0.7:  # double_Kelly_factor = 0.5
            prob_win = 0.5
        else:  # double_Kelly_factor = 0.35
            prob_win = 0.45
        return prob_win

    def dependence_count(self, MG):
        """check 0%"""
        if MG == 0:
            self.dependence_count_A += 1
            self.total_dependence += 1
        if MG == 1:
            self.dependence_count_B += 1
            self.total_dependence += 1
        if MG == 2:
            self.dependence_count_C += 1
            self.total_dependence += 1
        if MG == 3:
            self.dependence_count_D += 1
            self.total_dependence += 1

    def get_states_agent(self, df, price):
        """get states from data frame"""
        x = np.array(df.transpose())
        temp = np.array([])

        for i in range(Agent_N * 4):
            temp = np.append(temp, x[i][self.current_step])
        temp_gen = np.reshape(temp, [Agent_N, 4])

        for i in range(Agent_N):
            self.agent[i][0] = np.round((temp_gen[i][0] - temp_gen[i][1]) / W_max[i] + self.ESS[i], 3)
            if self.agent[i][0] < 0:  # 현재의 ESS 가 음수이므로 다음의 거래를 위해 일단 발전소가 현재 (-)량을 채움.
                self.dependence_count(i)
                self.agent[i][0] = 0  # 발전소로부터 에너지를 채우고 다시 0부터 시작

            if self.agent[i][0] > 1:
                self.agent[i][0] = 1

            self.agent[i][2] = price[i]  # price
            self.agent[i][3] = temp_gen[i][2]  # Month
            self.agent[i][4] = temp_gen[i][3]  # next_weather

            prob_win = self.cal_prob_win(self.agent[i][4])
            double_Kelly_factor = self.cal_kelly(prob_win)

            self.agent[i][1] = np.round((self.agent[i][0] ** 2) * double_Kelly_factor, 2)  # 거래되는 trading 량

        return self.agent

    def count_trading_energy(self, energy):
        self.count_energy += energy

    def each_step_count_energy(self):
        self.count_energy_step.append(self.count_energy)

    def battery_cycle_calculation(self):
        Lt_A = 0
        Lt_B = 0
        Lt_C = 0
        Lt_D = 0
        battery_MG_A, battery_MG_B, battery_MG_C, battery_MG_D = self.battery_get_L_calculation()

        for i in range(MAX_STEP):
            Lt_A += np.abs(battery_MG_A[i] - battery_MG_A[i + 1]) / 2

        for i in range(MAX_STEP):
            Lt_B += np.abs(battery_MG_B[i] - battery_MG_B[i + 1]) / 2

        for i in range(MAX_STEP):
            Lt_C += np.abs(battery_MG_C[i] - battery_MG_C[i + 1]) / 2

        for i in range(MAX_STEP):
            Lt_D += np.abs(battery_MG_D[i] - battery_MG_D[i + 1]) / 2

        return Lt_A, Lt_B, Lt_C, Lt_D

    def battery_get_L_calculation(self):
        beta_zero = 2731.7
        beta_one = 0.674
        beta_two = 1.614

        battery_MG_A = []
        battery_MG_B = []
        battery_MG_C = []
        battery_MG_D = []

        for i in range(MAX_STEP + 1):
            DOD = 1 - self.MG_A[i]

            if DOD == 0:
                CL = 2700
                L = 1 / CL
            else:
                CL = beta_zero * (DOD ** (- beta_one)) * np.exp(beta_two * self.MG_A[i])
                L = 1 / CL

            battery_MG_A.append(L)

        for i in range(MAX_STEP + 1):
            DOD = 1 - self.MG_B[i]

            if DOD == 0:
                CL = 2700
                L = 1 / CL
            else:
                CL = beta_zero * (DOD ** (- beta_one)) * np.exp(beta_two * self.MG_B[i])
                L = 1 / CL

            battery_MG_B.append(L)

        for i in range(MAX_STEP + 1):
            DOD = 1 - self.MG_C[i]

            if DOD == 0:
                CL = 2700
                L = 1 / CL
            else:
                CL = beta_zero * (DOD ** (- beta_one)) * np.exp(beta_two * self.MG_C[i])
                L = 1 / CL

            battery_MG_C.append(L)

        for i in range(MAX_STEP + 1):
            DOD = 1 - self.MG_D[i]

            if DOD == 0:
                CL = 2700
                L = 1 / CL
            else:
                CL = beta_zero * (DOD ** (- beta_one)) * np.exp(beta_two * self.MG_D[i])
                L = 1 / CL

            battery_MG_D.append(L)

        return battery_MG_A, battery_MG_B, battery_MG_C, battery_MG_D

    def input_mean_score(self, answer):
        self.reward_mean = answer

    def save_measurements(self):
        answer = np.mean(self.reward_mean)
        self.input_mean_score(answer)
        self.save_count_energy()
        self.save_ESS()
        self.save_price()

    def save_count_energy(self):
        np.save(ROUTE_MEASURE + 'step_count_energy.npy', self.count_energy_step)

    @staticmethod
    def save_episodes_count_energy(total):
        np.save(ROUTE_MEASURE + 'episodes_count_energy.npy', total)

    @staticmethod
    def save_sustain_factor(MG_A, MG_B, MG_C, MG_D):
        np.save(ROUTE_MEASURE + 'ESS_sustain_A.npy', MG_A)
        np.save(ROUTE_MEASURE + 'ESS_sustain_B.npy', MG_B)
        np.save(ROUTE_MEASURE + 'ESS_sustain_C.npy', MG_C)
        np.save(ROUTE_MEASURE + 'ESS_sustain_D.npy', MG_D)

    @staticmethod
    def save_dependence_factor(MG_A, MG_B, MG_C, MG_D):
        np.save(ROUTE_MEASURE + 'main_dependence_A.npy', MG_A)
        np.save(ROUTE_MEASURE + 'main_dependence_B.npy', MG_B)
        np.save(ROUTE_MEASURE + 'main_dependence_C.npy', MG_C)
        np.save(ROUTE_MEASURE + 'main_dependence_D.npy', MG_D)

    @staticmethod
    def save_battery_life(MG_A, MG_B, MG_C, MG_D):
        np.save(ROUTE_MEASURE + 'battery_cycle_A.npy', MG_A)
        np.save(ROUTE_MEASURE + 'battery_cycle_B.npy', MG_B)
        np.save(ROUTE_MEASURE + 'battery_cycle_C.npy', MG_C)
        np.save(ROUTE_MEASURE + 'battery_cycle_D.npy', MG_D)

    def save_ESS(self):
        np.save(ROUTE_SAVE + 'ESS_MG_A.npy', self.MG_A)
        np.save(ROUTE_SAVE + 'ESS_MG_B.npy', self.MG_B)
        np.save(ROUTE_SAVE + 'ESS_MG_C.npy', self.MG_C)
        np.save(ROUTE_SAVE + 'ESS_MG_D.npy', self.MG_D)

    def save_price(self):
        np.save(ROUTE_SAVE + 'price_MG_A.npy', self.price_plt_MG_A)
        np.save(ROUTE_SAVE + 'price_MG_B.npy', self.price_plt_MG_B)
        np.save(ROUTE_SAVE + 'price_MG_C.npy', self.price_plt_MG_C)
        np.save(ROUTE_SAVE + 'price_MG_D.npy', self.price_plt_MG_D)

    @staticmethod
    def save_mean_score(mean_score):
        np.save(ROUTE_SCORE + 'score_mean.npy', mean_score)

    def save_total_score(self, score):
        np.save(ROUTE_SCORE + 'score.npy', score)

    def save_ESS_week(self):
        temp_A = 0
        temp_B = 0
        temp_C = 0
        temp_D = 0
        for v in range(0, MAX_STEP + 1):
            temp_A += self.MG_A[v]
            temp_B += self.MG_B[v]
            temp_C += self.MG_C[v]
            temp_D += self.MG_D[v]
            if (v % 7 == 0) and (v != 0):
                self.MG_A_week.append(temp_A / 7)
                self.MG_B_week.append(temp_B / 7)
                self.MG_C_week.append(temp_C / 7)
                self.MG_D_week.append(temp_D / 7)
                temp_A = 0
                temp_B = 0
                temp_C = 0
                temp_D = 0
        fig = plt.figure()
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.plot(self.MG_A_week, 'g-', drawstyle='steps-post')
        plt.title('ESS_MG_1(SmartBuilding_A)')
        plt.xlim(0, x_max)
        plt.ylim(0, 100)
        plt.ylabel('ESS(%)')

        ax2 = fig.add_subplot(4, 1, 2)
        ax2.plot(self.MG_B_week, 'r-', drawstyle='steps-post')
        plt.title('ESS_MG_2(SmartBuilding_B)')
        plt.xlim(0, x_max)
        plt.ylim(0, 100)
        plt.xlabel('week')
        plt.ylabel('ESS(%)')

        ax3 = fig.add_subplot(4, 1, 3)
        ax3.plot(self.MG_C_week, 'b-', drawstyle='steps-post')
        plt.title('ESS_MG_3(SmartBuilding_C)')
        plt.xlim(0, x_max)
        plt.ylim(0, 100)
        plt.xlabel('week')
        plt.ylabel('ESS(%)')

        ax4 = fig.add_subplot(4, 1, 4)
        ax4.plot(self.MG_D_week, 'y-', drawstyle='steps-post')
        plt.title('ESS_MG_4(SmartBuilding_D)')
        plt.xlim(0, x_max)
        plt.ylim(0, 100)
        plt.xlabel('week')
        plt.ylabel('ESS(%)')
        plt.tight_layout()
        plt.savefig(ROUTE_SAVE + 'save_ESS_week.png', dpi=500)
        plt.close(fig)
        del self.MG_A_week[:]
        del self.MG_B_week[:]
        del self.MG_C_week[:]
        del self.MG_D_week[:]

    def save_price_week(self):
        temp_price_A = 0
        temp_price_B = 0
        temp_price_C = 0
        temp_price_D = 0
        for s in range(0, MAX_STEP +1):
            temp_price_A += self.price_plt_MG_A[s]
            temp_price_B += self.price_plt_MG_A[s]
            temp_price_C += self.price_plt_MG_A[s]
            temp_price_D += self.price_plt_MG_A[s]
            if (s % 7 == 0) and (s != 0):
                self.price_plt_MG_A.append(temp_price_A / 7)
                self.price_plt_MG_B.append(temp_price_B / 7)
                self.price_plt_MG_C.append(temp_price_C / 7)
                self.price_plt_MG_D.append(temp_price_D / 7)
                temp_price_A = 0
                temp_price_B = 0
                temp_price_C = 0
                temp_price_D = 0
        fig = plt.figure()
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.plot(self.price_plt_MG_A, 'g-', drawstyle='steps-post')
        plt.title('Price_MG_1(SmartBuilding_A)')
        plt.xlim(0, x_max)
        plt.ylim(standard_price[0] * 1.1, standard_price[0] * 1.39)
        plt.xlabel('week')
        plt.ylabel('price(won/KWatt)')

        ax2 = fig.add_subplot(4, 1, 2)
        ax2.plot(self.price_plt_MG_B, 'r-', drawstyle='steps-post')
        plt.title('Price_MG_2(SmartBuilding_B)')
        plt.xlim(0, x_max)
        plt.ylim(standard_price[1] * 1.1, standard_price[1] * 1.39)
        plt.xlabel('week')
        plt.ylabel('price(won/KWatt)')

        ax3 = fig.add_subplot(4, 1, 3)
        ax3.plot(self.price_plt_MG_C, 'b-', drawstyle='steps-post')
        plt.title('Price_MG_3(SmartBuilding_C)')
        plt.xlim(0, x_max)
        plt.ylim(standard_price[2] * 1.1, standard_price[2] * 1.39)
        plt.xlabel('week')
        plt.ylabel('price(won/KWatt)')

        ax4 = fig.add_subplot(4, 1, 4)
        ax4.plot(self.price_plt_MG_D, 'y-', drawstyle='steps-post')
        plt.title('Price_MG_4(SmartBuilding_D)')
        plt.xlim(0, x_max)
        plt.ylim(standard_price[3] * 1.1, standard_price[3] * 1.39)
        plt.xlabel('week')
        plt.ylabel('price(won/KWatt)')
        plt.tight_layout()
        plt.savefig(ROUTE_SAVE + 'save_price_week.png', dpi=500)
        plt.close(fig)
        del self.price_plt_MG_A[:]
        del self.price_plt_MG_B[:]
        del self.price_plt_MG_C[:]
        del self.price_plt_MG_D[:]

    # measurement factor 3, dependence, sustain, battery_cycle
    def return_dependence_factor(self):
        dependence = np.array([self.dependence_count_A, self.dependence_count_B, self.dependence_count_C,
                               self.dependence_count_D])
        return dependence

    def return_sustain_factor(self):
        sustain = np.array([self.sustain_count_A, self.sustain_count_B, self.sustain_count_C, self.sustain_count_D])
        return sustain

    def return_battery_cycle_factor(self):
        A, B, C, D = self.battery_cycle_calculation()
        A = np.round(A, 5)
        B = np.round(B, 5)
        C = np.round(C, 5)
        D = np.round(D, 5)

        if A == 0:
            A = 100000
        else:
            A = 1/A

        if B == 0:
            B = 100000
        else:
            B = 1/B

        if C == 0:
            C = 100000
        else:
            C = 1/C

        if D == 0:
            D = 100000
        else:
            D = 1/D

        battery = np.array([A, B, C, D])
        return battery

    def return_mean_score(self):
        return self.reward_mean

    def return_count_energy(self):
        return self.count_energy

    @staticmethod
    def render_count_energy():
        MG = np.load(ROUTE_MEASURE + 'episodes_count_energy.npy')
        plt.plot(MG)
        plt.xlabel('episode_num')
        plt.ylabel('energy')
        plt.title('MG')
        plt.show()

    @staticmethod
    def render_mean_score():
        mean_score = np.load(ROUTE_SCORE + 'score_mean.npy')
        plt.figure()
        plt.plot(mean_score)
        plt.show()

    @staticmethod
    def render_sustain():
        fig1 = plt.figure()
        MG_A = np.load(ROUTE_MEASURE + 'ESS_sustain_A.npy')
        MG_B = np.load(ROUTE_MEASURE + 'ESS_sustain_B.npy')
        MG_C = np.load(ROUTE_MEASURE + 'ESS_sustain_C.npy')
        MG_D = np.load(ROUTE_MEASURE + 'ESS_sustain_D.npy')

        plt.plot(MG_A, 'r-')
        plt.plot(MG_B, 'y-')
        plt.plot(MG_C, 'g-')
        plt.plot(MG_D, 'b-')

        plt.legend(['MG_1', 'MG_2', 'MG_3', 'MG_4'])
        plt.xlabel('episode_num')
        plt.ylabel('sustain_num')
        plt.title('MGs_sustain')
        plt.show()

        plt.close(fig1)

    @staticmethod
    def render_dependence():
        fig2 = plt.figure()
        MG_A = np.load(ROUTE_MEASURE + 'main_dependence_A.npy')
        MG_B = np.load(ROUTE_MEASURE + 'main_dependence_B.npy')
        MG_C = np.load(ROUTE_MEASURE + 'main_dependence_C.npy')
        MG_D = np.load(ROUTE_MEASURE + 'main_dependence_D.npy')

        plt.plot(MG_A, 'r-')
        plt.plot(MG_B, 'y-')
        plt.plot(MG_C, 'g-')
        plt.plot(MG_D, 'b-')

        plt.legend(['MG_1', 'MG_2', 'MG_3', 'MG_4'])
        plt.xlabel('episode_num')
        plt.ylabel('dependence_num')
        plt.title('MGs_dependence')
        plt.show()

        plt.close(fig2)

    @staticmethod
    def render_battery_cycle():
        fig3 = plt.figure()
        MG_A = np.load(ROUTE_MEASURE + 'battery_cycle_A.npy')
        MG_B = np.load(ROUTE_MEASURE + 'battery_cycle_B.npy')
        MG_C = np.load(ROUTE_MEASURE + 'battery_cycle_C.npy')
        MG_D = np.load(ROUTE_MEASURE + 'battery_cycle_D.npy')

        plt.plot(MG_A, 'r-')
        plt.plot(MG_B, 'y-')
        plt.plot(MG_C, 'g-')
        plt.plot(MG_D, 'b-')

        plt.legend(['MG_1', 'MG_2', 'MG_3', 'MG_4'])
        plt.xlabel('episode_num')
        plt.ylabel('left_battery_cycle_num')
        plt.title('MG_battery_cycle')
        plt.show()

        plt.close(fig3)

    # 상태 초기화
    def reset(self):
        self.count_energy_step = []
        self.count_energy = 0
        self.dependence_count_A = 0
        self.dependence_count_B = 0
        self.dependence_count_C = 0
        self.dependence_count_D = 0
        self.total_dependence = 0
        self.sustain_count_A = 0
        self.sustain_count_B = 0
        self.sustain_count_C = 0
        self.sustain_count_D = 0
        self.total_sustain = 0
        self.ESS_reward = 0
        self.save_step = np.array([])
        self.save_reward = np.array([])
        self.save_action = np.array([])
        self.save_ESS_1 = np.array([])
        self.save_ESS_2 = np.array([])
        self.save_ESS_3 = np.array([])
        self.save_ESS_4 = np.array([])
        self.load_step = []
        self.load_reward = []
        self.load_action = []
        self.load_ESS_1 = []
        self.load_ESS_2 = []
        self.load_ESS_3 = []
        self.load_ESS_4 = []
        self.reward_mean = np.array([])
        self.reward_mean = 0
        self.reward = 0
        self.current_step = 0
        self.price_ratio = np.random.rand(Agent_N, 1)
        self.ESS = np.array([0.2, 0.2, 0.2, 0.2])
        self.price = np.random.rand(Agent_N, 1)
        self.cur_states = np.random.rand(Agent_N, 5)
        self.agent = np.random.rand(Agent_N, 5)
        self.total_reward = 0
        self.MG_A = []
        self.MG_B = []
        self.MG_C = []
        self.MG_D = []
        self.MG_A_week = []
        self.MG_B_week = []
        self.MG_C_week = []
        self.MG_D_week = []
        self.score_plt = []
        self.score_plt_week = []
        self.price_plt_MG_A = []
        self.price_plt_MG_B = []
        self.price_plt_MG_C = []
        self.price_plt_MG_D = []
        self.score_plt_ep = []

        init_price = standard_price  # MG's 의 kw 당 원가

        self.cur_states = self.get_states_agent(self.df, init_price)

        for i in range(Agent_N):
            self.price[i] = self.cur_states[i][2]

        self.obs = np.reshape(self.cur_states, (1, Agent_N * 5))

        return self.obs