import numpy as np
import pandas as pd
import gym
from gym import spaces
from parameters import dg_parameters, ev_parameters


class Constant:  # 定义一个名为 Constant 的类，用于存储常量数据
    MONTHS_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 每个月的天数，常用于日期计算
    MAX_STEP_HOURS = 24 * 30  # 表示一个月内最大的小时数，用于时间步长的计算


class DataManager:  # 定义一个名为 DataManager 的类，用于管理数据
    def __init__(self) -> None:
        self.PV_Generation = []  # 初始化一个列表，用于存储光伏发电数据
        self.Prices = []  # 初始化一个列表，用于存储电价数据
        self.Electricity_Consumption = []  # 初始化一个列表，用于存储电力消耗数据

    def add_pv_element(self, element):  # 定义一个方法，用于向 PV_Generation 列表中添加数据
        self.PV_Generation.append(element)  # 将参数 element 添加到 PV_Generation 列表中

    def add_price_element(self, element):  # 定义一个方法，用于向 Prices 列表中添加数据
        self.Prices.append(element)  # 将参数 element 添加到 Prices 列表中

    def add_electricity_element(self, element):  # 定义一个方法，用于向 Electricity_Consumption 列表中添加数据
        self.Electricity_Consumption.append(element)  # 将参数 element 添加到 Electricity_Consumption 列表中

    # 定义一个方法，根据提供的月份、日期和小时获取光伏发电数据
    def get_pv_data(self, month, day, day_time):
        return self.PV_Generation[
            (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 24 + day_time]  # 从 PV_Generation 列表中获取对应时间点的数据

    # 定义一个方法，根据提供的月份、日期和小时获取电价数据
    def get_price_data(self, month, day, day_time):
        return self.Prices[
            (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 24 + day_time]  # 从 Prices 列表中获取对应时间点的数据

    # 定义一个方法，根据提供的月份、日期和小时获取电力消耗数据
    def get_electricity_cons_data(self, month, day, day_time):
        return self.Electricity_Consumption[
            (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 24 + day_time]  # 从 Electricity_Consumption 列表中获取对应时间点的数据

    # 定义一个方法，获取一天内的连续光伏发电数据
    def get_series_pv_data(self, month, day):
        return self.PV_Generation[
               (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 24:(sum(Constant.MONTHS_LEN[
                                                                          :month - 1]) + day - 1) * 24 + 24]

    # 定义一个方法，获取一天内的连续电价数据
    def get_series_price_data(self, month, day):
        return self.Prices[
               (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 24:(sum(Constant.MONTHS_LEN[
                                                                          :month - 1]) + day - 1) * 24 + 24]

    # 定义一个方法，获取一天内的连续电力消耗数据
    def get_series_electricity_cons_data(self, month, day):
        return self.Electricity_Consumption[
               (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 24:(sum(Constant.MONTHS_LEN[
                                                                          :month - 1]) + day - 1) * 24 + 24]


class EV:
    def __init__(self, parameters):
        self.capacity = parameters['capacity']
        self.max_charge = parameters['max_charge']
        self.max_discharge = parameters['max_discharge']
        self.efficiency = parameters['efficiency']
        self.initial_capacity = parameters['initial_capacity']
        self.min_soc = parameters['min_soc']
        self.max_soc = parameters['max_soc']
        self.current_capacity = self.initial_capacity
        self.degradation = 0  # 电池退化成本
        self.energy_change = 0

    def step(self, action_ev):
        energy = action_ev * self.max_charge  # 计算根据动作决定的能量变化
        updated_capacity = max(self.min_soc,
                               min(self.max_soc, (self.current_capacity * self.capacity + energy) / self.capacity))
        # 更新电池容量，确保它在最小和最大荷电状态之间
        self.energy_change = (updated_capacity - self.current_capacity) * self.capacity  # 计算能量变化量
        self.current_capacity = updated_capacity  # 更新当前容量

    def _get_cost(self, current_price, energy_change):
        cost = current_price * energy_change
        return cost

    def SOC(self):  # 获取电池的当前荷电状态
        return self.current_capacity

    def reset(self):
        self.current_capacity = np.random.uniform(0.2, 0.8)


class DG:
    def __init__(self, parameters):
        self.name = parameters.keys()  # 保存传入参数字典的键作为属性名
        self.a_factor = parameters['a']  # 从参数中获取 a 因子，用于计算成本函数
        self.b_factor = parameters['b']  # 从参数中获取 b 因子，用于计算成本函数
        self.c_factor = parameters['c']  # 从参数中获取 c 因子，用于计算成本函数
        self.power_output_max = parameters['power_output_max']  # 设置发电机的最大输出功率
        self.power_output_min = parameters['power_output_min']  # 设置发电机的最小输出功率
        self.ramping_up = parameters['ramping_up']  # 设置功率增加的最大速率
        self.ramping_down = parameters['ramping_down']  # 设置功率减少的最大速率
        self.current_output = 0
        self.last_step_output = None  # 初始化上一步的输出为 None

    def step(self, action_gen):
        output_change = action_gen * self.ramping_up  # 计算功率的变化量，基于动作乘以增加速率
        output = self.current_output + output_change  # 更新输出功率
        if output > 0:
            output = max(self.power_output_min, min(self.power_output_max, output))  # 确保输出在最小和最大功率范围内
        else:
            output = 0  # 如果计算结果小于等于零，则输出设置为0
        self.current_output = output  # 更新当前输出

    def _get_cost(self, output):
        if output <= 0:
            cost = 0  # 如果输出为0或负值，则成本为0
        else:
            cost = (self.a_factor * pow(output, 2) + self.b_factor * output + self.c_factor)  # 计算输出成本
        return cost  # 返回成本值

    def reset(self):
        self.current_output = 0  # 重置发电机输出为0


class Grid:  # 定义一个名为 Grid 的类，用于管理与电网相关的操作和属性
    def __init__(self):
        self.on = True  # 将电网的开启状态初始化为 True，表示电网默认为开启状态
        if self.on:
            self.exchange_ability = 100  # 如果电网开启，设置交换能力为 100
        else:
            self.exchange_ability = 0  # 如果电网关闭，设置交换能力为 0
        self.day = 0  # 初始化 day 属性，用于跟踪当前是月份中的第几天
        self.time = 0  # 初始化 time 属性，用于跟踪一天中的时间（可能是小时数）
        self.past_price = []  # 初始化 past_price 属性，用于存储过去的电价数据
        self.price = []  # 初始化 price 属性，用于存储当前的电价数据

    def _get_cost(self, current_price, energy_exchange):
        return current_price * energy_exchange  # 计算能量交换的成本，成本等于当前电价乘以交换的能量

    def retrive_past_price(self):
        result = []  # 初始化一个空列表，用于存储返回的过去价格数据
        if self.day < 1:
            past_price = self.past_price  # 如果当前 day 小于1，使用 past_price 列表作为过去价格数据源
        else:
            past_price = self.price[24 * (self.day - 1):24 * self.day]  # 否则，从 price 列表中获取当前 day 对应的24小时的价格数据
        for item in past_price[(self.time - 24)::]:
            result.append(item)  # 从 past_price 中获取从当前时间向前推24小时的价格数据，添加到结果列表中
        for item in self.price[24 * self.day:(24 * self.day + self.time)]:
            result.append(item)  # 获取从当天开始到当前时间的价格数据，添加到结果列表中
        return result  # 返回收集到的过去价格数据列表


class ESSEnv(gym.Env):  # 定义一个名为 ESSEnv 的类，继承自 gym 环境，用于强化学习模拟
    def __init__(self, **kwargs):  # 类的构造函数，接收可变数量的关键字参数
        super(ESSEnv, self).__init__()  # 调用父类的构造函数，初始化基类
        # parameters
        self.data_manager = DataManager()  # 创建 DataManager 的实例，用于管理和存储数据
        self._load_year_data()  # 调用方法加载一年的数据
        self.episode_length = kwargs.get('episode_length', 24)  # 获取关键字参数指定的一个episode的长度，默认为24小时
        self.month = None  # 初始化月份为 None，用于在环境中追踪当前月份 None表示从0开始
        self.day = None  # 初始化天数为 None，用于在环境中追踪当前日期
        self.TRAIN = True  # 设置环境为训练模式
        self.current_time = None  # 初始化当前时间为 None
        self.dg_parameters = kwargs.get('dg_parameters', dg_parameters)  # 获取柴油发电机参数，默认值在代码外部定义
        self.ev_parameters = kwargs.get('ev_parameters', ev_parameters)
        self.penalty_coefficient = 50  # 设置惩罚系数，用于计算软约束的惩罚
        self.sell_coefficient = 0.5  # 设置销售系数，用于计算售电收益
        self.grid = Grid()  # 创建 Grid 的实例，用于管理电网相关操作
        self.dg1 = DG(self.dg_parameters['gen_1'])  # 根据提供的参数创建第一个柴油发电机的实例
        self.dg2 = DG(self.dg_parameters['gen_2'])  # 根据提供的参数创建第二个柴油发电机的实例
        self.dg3 = DG(self.dg_parameters['gen_3'])  # 根据提供的参数创建第三个柴油发电机的实例
        self.ev = EV(self.ev_parameters)  # 创建 EV 的实例，用于管理电动车相关操作
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # 定义动作空间，这里是5维的连续空间，每个动作的值介于-1和1之间
        self.state_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)  # 定义状态空间，这里是7维的连续空间，每个状态的值介于0和1之间

    @property
    def netload(self):
        return self.demand - self.grid.wp_gen - self.grid.pv_gen

    def reset(self, ):
        self.month = np.random.randint(1, 13)  # 随机选择一个月份（1到12月）
        if self.TRAIN:
            self.day = np.random.randint(1, 20)  # 如果是训练模式，随机选择一个月中的前20天
        else:
            self.day = np.random.randint(20, Constant.MONTHS_LEN[self.month] - 1)  # 如果不是训练模式，随机选择一个月中的后几天
        self.current_time = 0  # 将当前时间重置为0
        self.dg1.reset()  # 重置第一个柴油发电机
        self.dg2.reset()  # 重置第二个柴油发电机
        self.dg3.reset()  # 重置第三个柴油发电机
        self.ev.reset()
        return self._build_state()  # 构建并返回当前状态

    def _build_state(self):
        ev_soc = self.ev.SOC()
        dg1_output = self.dg1.current_output  # 获取第一个柴油发电机的当前输出
        dg2_output = self.dg2.current_output  # 获取第二个柴油发电机的当前输出
        dg3_output = self.dg3.current_output  # 获取第三个柴油发电机的当前输出
        ev_output = self.ev.energy_change
        time_step = self.current_time  # 获取当前时间步
        electricity_demand = self.data_manager.get_electricity_cons_data(self.month, self.day,
                                                                         self.current_time)  # 获取当前的电力需求数据
        pv_generation = self.data_manager.get_pv_data(self.month, self.day, self.current_time)  # 获取当前的光伏发电数据
        price = self.data_manager.get_price_data(self.month, self.day, self.current_time)  # 获取当前的电价
        net_load = electricity_demand - pv_generation  # 计算净负载
        obs = np.concatenate((np.float32(time_step), np.float32(price), np.float32(net_load),
                              np.float32(dg1_output), np.float32(dg2_output), np.float32(dg3_output),
                              np.float32(ev_output), np.float32(ev_soc)),
                             axis=None)  # 将所有状态信息合并为一个观测数组
        return obs  # 返回观测数组，通常用作强化学习算法的输入

    def step(self, action):  # 环境中执行一个动作，返回新的观察、奖励和是否结束
        current_obs = self._build_state()  # 构建当前状态
        self.dg1.step(action[0])  # 执行第一个柴油发电机的状态转换
        self.dg2.step(action[1])  # 执行第二个柴油发电机的状态转换
        self.dg3.step(action[2])  # 执行第三个柴油发电机的状态转换
        self.ev.step(action[3])  # 更新 EV 状态

        current_output = np.array((self.dg1.current_output, self.dg2.current_output, self.dg3.current_output,
                                   -self.ev.energy_change))  # 创建一个包含所有发电机输出和电池能量变化的数组
        self.current_output = current_output  # 更新当前输出
        actual_production = sum(current_output)  # 计算总实际生产量
        netload = current_obs[3]  # 从当前观察中获取净负载
        price = current_obs[1]  # 从当前观察中获取电价

        unbalance = actual_production - netload  # 计算不平衡量（实际生产与需求之差）

        reward = 0
        excess_penalty = 0
        deficient_penalty = 0
        sell_benefit = 0
        buy_cost = 0
        self.excess = 0
        self.shedding = 0

        if unbalance >= 0:  # 如果产生过剩
            if unbalance <= self.grid.exchange_ability:
                sell_benefit = self.grid._get_cost(price, unbalance) * self.sell_coefficient  # 计算向电网出售电力的收益
            else:
                sell_benefit = self.grid._get_cost(price, self.grid.exchange_ability) * self.sell_coefficient
                self.excess = unbalance - self.grid.exchange_ability  # 计算超出电网交换能力的过剩量
                excess_penalty = self.excess * self.penalty_coefficient  # 计算过剩惩罚
        else:  # 如果电力不足
            if abs(unbalance) <= self.grid.exchange_ability:
                buy_cost = self.grid._get_cost(price, abs(unbalance))  # 从电网购买不足量的电力的成本
            else:
                buy_cost = self.grid._get_cost(price, self.grid.exchange_ability)
                self.shedding = abs(unbalance) - self.grid.exchange_ability  # 计算电网也无法提供的不足量
                deficient_penalty = self.shedding * self.penalty_coefficient  # 计算不足惩罚
        dg1_cost = self.dg1._get_cost(self.dg1.current_output)
        dg2_cost = self.dg2._get_cost(self.dg2.current_output)
        dg3_cost = self.dg3._get_cost(self.dg3.current_output)
        ev_cost = self.ev._get_cost(price, self.ev.energy_change)  # 计算 EV 成本
        #ev_buy=
        #ev_sell=
        reward -= (dg1_cost + dg2_cost + dg3_cost + ev_cost + excess_penalty +
                   deficient_penalty - sell_benefit + buy_cost) / 1e3  # 计算总奖励
        self.operation_cost = ev_cost + dg1_cost + dg2_cost + dg3_cost + buy_cost - sell_benefit + excess_penalty + deficient_penalty
        self.unbalance = unbalance
        self.real_unbalance = self.shedding + self.excess
        final_step_outputs = [self.dg1.current_output, self.dg2.current_output, self.dg3.current_output,
                              self.ev.energy_change]  # 创建最终的输出状态列表
        self.current_time += 1  # 更新当前时间步
        finish = (self.current_time == self.episode_length)  # 判断episode是否结束

        if finish:  # 如果episode结束
            self.final_step_outputs = final_step_outputs  # 保存最终输出状态
            self.current_time = 0  # 重置当前时间为0
            # 下面的注释代码可能用于日期的更新，但在这里被注释掉了
            # self.day+=1
            # if self.day>Constant.MONTHS_LEN[self.month-1]:
            #     self.day=1
            #     self.month+=1
            # if self.month>12:
            #     self.month=1
            #     self.day=1
            next_obs = self.reset()  # 重置环境，获取新的观察
        else:
            next_obs = self._build_state()  # 如果episode未结束，构建下一步的状态

        return current_obs, next_obs, float(reward), finish  # 返回当前观察、下一观察、奖励和是否结束的标志

    def render(self, current_obs, next_obs, reward, finish):  # render 方法用于输出环境的当前状态，方便调试和观察强化学习算法的行为。
        print('day={},hour={:2d}, state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(self.day,
                                                                                                self.current_time,
                                                                                                current_obs, next_obs,
                                                                                                reward, finish))

    def _load_year_data(self):
        pv_df = pd.read_csv('PV.csv', sep=';')  # 从CSV文件加载光伏发电数据
        price_df = pd.read_csv('Prices.csv', sep=';')  # 从CSV文件加载电价数据
        electricity_df = pd.read_csv('H4.csv', sep=';')  # 从CSV文件加载电力消费数据

        pv_data = pv_df['P_PV_'].apply(lambda x: x.replace(',', '.')).to_numpy(dtype=float)  # 将光伏数据中的逗号替换为点，并转换为浮点数数组
        price = price_df['Price'].apply(lambda x: x.replace(',', '.')).to_numpy(dtype=float)  # 将价格数据中的逗号替换为点，并转换为浮点数数组
        electricity = electricity_df['Power'].apply(lambda x: x.replace(',', '.')).to_numpy(
            dtype=float)  # 将电力消费数据中的逗号替换为点，并转换为浮点数数组

        # 添加处理后的光伏数据到数据管理器
        for element in pv_data:
            self.data_manager.add_pv_element(element * 200)  # 将数据按比例放大并添加

        # 添加处理后的价格数据到数据管理器，同时进行条件处理
        for element in price:
            element /= 10  # 将价格数据按比例缩小
            if element <= 0.5:
                element = 0.5  # 如果价格低于0.5，则设为0.5
            self.data_manager.add_price_element(element)  # 添加处理后的价格数据

        # 每60分钟的电力消费数据进行累加后添加到数据管理器
        for i in range(0, electricity.shape[0], 60):
            element = electricity[i:i + 60]  # 获取每小时的数据
            self.data_manager.add_electricity_element(sum(element) * 300)  # 将数据按比例放大并添加


if __name__ == '__main__':
    env = ESSEnv()  # 创建环境的实例
    env.TRAIN = False  # 设置环境为非训练模式，可能影响某些内部逻辑，如数据选择或行为
    rewards = []  # 初始化一个列表来存储每一步的奖励
    current_obs = env.reset()  # 重置环境到初始状态，获取初始观察
    tem_action = [0.1, 0.1, 0.1, 0.1]  # 定义一个临时动作，用于测试环境反应

    for _ in range(144):  # 进行144步模拟，假设每天144个时间步
        print(f'current month is {env.month}, current day is {env.day}, current time is {env.current_time}')
        current_obs, next_obs, reward, finish = env.step(tem_action)  # 执行一个步骤，应用动作
        env.render(current_obs, next_obs, reward, finish)  # 调用 render 方法显示当前步骤的结果
        current_obs = next_obs  # 更新当前观察为新的观察
        rewards.append(reward)  # 将这一步的奖励添加到奖励列表
