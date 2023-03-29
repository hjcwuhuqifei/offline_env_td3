"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pyglet
from math import sqrt
import pickle
from pathlib import Path
import random


def get_reward(observation, terminal, collision, action, reach):
    if collision:
        reward = action/15  - collision * 20
    else:
         reward = action/15 +  terminal * 20 * reach

    return reward


def normalize_angle(angle_rad):
    # to normalize an angle to [-pi, pi]
    a = math.fmod(angle_rad + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a = a + 2.0 * math.pi
    return a - math.pi


def linear_interpolate(path_point_0, path_point_1, rs_inter):
    ''' path point interpolated linearly according to rs value
    path_point_0 should be prior to path_point_1'''

    def lerp(x0, x1, w):
        return x0 + w * (x1 - x0)

    def slerp(a0, a1, w):
        # angular, for theta
        a0_n = normalize_angle(a0)
        a1_n = normalize_angle(a1)
        d = a1_n - a0_n
        if d > math.pi:
            d = d - 2 * math.pi
        elif d < -math.pi:
            d = d + 2 * math.pi
        a = a0_n + w * d
        return normalize_angle(a)

    rs_0 = path_point_0[2]
    rs_1 = path_point_1[2]
    weight = (rs_inter - rs_0) / (rs_1 - rs_0)
    if weight < 0 or weight > 1:
        print("weight error, not in [0, 1]")

    rx_inter = lerp(path_point_0[0], path_point_1[0], weight)
    ry_inter = lerp(path_point_0[1], path_point_1[1], weight)
    rtheta_inter = slerp(path_point_0[3], path_point_1[3], weight)
    return rx_inter, ry_inter, rtheta_inter


def object_to_ego(x, y, yaw):
    res_x = math.cos(yaw) * x - math.sin(yaw) * y
    res_y = math.sin(yaw) * x + math.cos(yaw) * y
    return res_x, res_y


class OfflineRL_test(gym.Env):
    def __init__(self):
        argoverse_scenario_dir = Path(
            '/home/haojiachen/桌面/offline_rl/offline_/')
        all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.pkl"))
        scenario_file_lists = (all_scenario_files[:112])
        self.scenarios = []
        for scenario_file_list in scenario_file_lists:
            scenario = pickle.load(open(scenario_file_list, 'rb'))
            self.scenarios.append(scenario)

        self.scenario = self.scenarios[0]
        self.index_of_scenario = 0
        self.ego_track = self.scenario['EGO']
        self.object_tracks = []
        for key in self.scenario.keys():
            if key != 'EGO' and key != 'states':
                self.object_tracks.append(self.scenario[key])

        self.ego_x = self.ego_track.object_states[0].position[0]
        self.ego_y = self.ego_track.object_states[0].position[1]
        self.ego_yaw = self.ego_track.object_states[0].heading
        self.ego_v = sqrt(self.ego_track.object_states[0].velocity[0] ** 2 +
                          self.ego_track.object_states[0].velocity[1] ** 2)
        self.trajectory = []
        s = 0
        for i in range(len(self.ego_track.object_states)):
            if i == 0:
                self.trajectory.append(
                    [self.ego_track.object_states[i].position[0], self.ego_track.object_states[i].position[1],
                     s, self.ego_track.object_states[i].heading])
            else:
                delta_s = sqrt((self.ego_track.object_states[i].position[0] -
                                self.ego_track.object_states[i - 1].position[0]) ** 2 +
                               (self.ego_track.object_states[i].position[1] -
                                self.ego_track.object_states[i - 1].position[1]) ** 2)
                s += delta_s
                self.trajectory.append(
                    [self.ego_track.object_states[i].position[0], self.ego_track.object_states[i].position[1],
                     s, self.ego_track.object_states[i].heading])

        self.time = 0
        self.dt = 0.1
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        self.x_threshold = 100000  # 小车x方向最大运动范围
        self.v_threshold = 100000
        self.max_a = 10
        high = np.array([self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold,
                         self.x_threshold,
                         self.x_threshold,
                         self.v_threshold
                         ],
                        dtype=np.float32)

        self.action_space = spaces.Box(
            low=-self.max_a,
            high=self.max_a, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        done = 0
        collision = 0
        reach = 0

        self.s += (action[0] + 10) * 0.1
        self.ego_v = action[0] + 10
        # 插值得到现在的x, y, yaw
        if self.s > self.trajectory[-1][2]:
            self.ego_x = self.trajectory[-1][0]
            self.ego_y = self.trajectory[-1][1]
            self.ego_yaw = self.trajectory[-1][3]
            reach = 1
            done = 1
        else:
            for i in range(1, len(self.trajectory)):
                if self.trajectory[i - 1][2] < self.s < self.trajectory[i][2]:
                    self.ego_x, self.ego_y, self.ego_yaw = linear_interpolate(self.trajectory[i - 1],
                                                                              self.trajectory[i],
                                                                              self.s)
                    break

        # 通过环境车辆获取当前观测值
        object_front = [0, 0, 0, 0]
        object_behind = [-0, 0, 0, 0]
        object_left_front = [0, 0, 0, 0]
        object_right_front = [0, -0, 0, 0]
        object_left_behind = [-0, 0, 0, 0]
        object_right_behind = [-0, -0, 0, 0]
        for else_car_track in self.object_tracks:
            # 时间框定
            if else_car_track.object_states[0].timestep <= self.time <= else_car_track.object_states[-1].timestep:
                for object_state in else_car_track.object_states:
                    if object_state.timestep == self.time:
                        object_x = object_state.position[0]
                        object_y = object_state.position[1]
                        object_yaw = object_state.heading
                        object_v_x = object_state.velocity[0]
                        object_v_y = object_state.velocity[1]

                        x_to_ego, y_to_ego = object_to_ego(object_x - self.ego_x, object_y - self.ego_y, -self.ego_yaw)

                        if abs(object_yaw - self.ego_yaw) > 90 / 180 * math.pi:
                            continue
                        if x_to_ego > 30 or x_to_ego < -12 or y_to_ego < -5.25 or y_to_ego > 5.25:
                            continue
                        # 选择前车
                        if 0 < x_to_ego and -1.75 <= y_to_ego <= 1.75:
                            if object_front[0] == 0:
                                object_front[0] = x_to_ego
                                object_front[1] = y_to_ego
                                object_front[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                                object_front[3] = 1
                            elif x_to_ego < object_front[0]:
                                object_front[0] = x_to_ego
                                object_front[1] = y_to_ego
                                object_front[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                            else:
                                continue
                        # 选择后车
                        if x_to_ego < 0 and -1.75 <= y_to_ego <= 1.75:
                            if  object_behind[0] == 0:
                                object_behind[0] = x_to_ego
                                object_behind[1] = y_to_ego
                                object_behind[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                                object_behind[3] = 1
                            elif x_to_ego > object_behind[0]:
                                object_behind[0] = x_to_ego
                                object_behind[1] = y_to_ego
                                object_behind[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                            else:
                                continue
                        # 选择右前车
                        if y_to_ego < -1.75 and x_to_ego >= 0:
                            if object_right_front[0] == 0 and object_right_front[1] == 0:
                                object_right_front[0] = x_to_ego
                                object_right_front[1] = y_to_ego
                                object_right_front[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                                object_right_front[3] = 1
                            elif math.sqrt(x_to_ego ** 2 + y_to_ego ** 2) < math.sqrt(
                                    object_right_front[0] ** 2 + object_right_front[1] ** 2):
                                object_right_front[0] = x_to_ego
                                object_right_front[1] = y_to_ego
                                object_right_front[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                            else:
                                continue
                        # 选择左前车
                        if y_to_ego > 1.75 and x_to_ego >= 0:
                            if object_left_front[0] == 0 and object_left_front[1] == 0:
                                object_left_front[0] = x_to_ego
                                object_left_front[1] = y_to_ego
                                object_left_front[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                                object_left_front[3] = 1
                            elif math.sqrt(x_to_ego ** 2 + y_to_ego ** 2) < math.sqrt(
                                    object_left_front[0] ** 2 + object_left_front[1] ** 2):
                                object_left_front[0] = x_to_ego
                                object_left_front[1] = y_to_ego
                                object_left_front[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                            else:
                                continue
                        # 选择右后车
                        if y_to_ego < -1.75 and x_to_ego < 0:
                            if object_right_behind[0] == 0 and object_right_behind[1] == 0:
                                object_right_behind[0] = x_to_ego
                                object_right_behind[1] = y_to_ego
                                object_right_behind[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                                object_right_behind[3] = 1
                            elif math.sqrt(x_to_ego ** 2 + y_to_ego ** 2) < math.sqrt(
                                    object_right_behind[0] ** 2 + object_right_behind[1] ** 2):
                                object_right_behind[0] = x_to_ego
                                object_right_behind[1] = y_to_ego
                                object_right_behind[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                            else:
                                continue
                        # 选择左后车
                        if y_to_ego > 1.75 and x_to_ego < 0:
                            if object_left_front[0] == 0 and object_left_front[1] == 0:
                                object_left_behind[0] = x_to_ego
                                object_left_behind[1] = y_to_ego
                                object_left_behind[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                                object_left_behind[3] = 1
                            elif math.sqrt(x_to_ego ** 2 + y_to_ego ** 2) < math.sqrt(
                                    object_left_behind[0] ** 2 + object_left_behind[1] ** 2):
                                object_left_behind[0] = x_to_ego
                                object_left_behind[1] = y_to_ego
                                object_left_behind[2] = sqrt(object_v_x ** 2 + object_v_y ** 2)
                            else:
                                continue
                    else:
                        continue
            else:
                continue

        observation = np.array([self.ego_v,
                       object_front[0], object_front[1], object_front[2],
                       object_behind[0], object_behind[1], object_behind[2],
                       object_left_front[0], object_left_front[1], object_left_front[2],
                       object_right_front[0], object_right_front[1], object_right_front[2],
                       object_left_behind[0], object_left_behind[1], object_left_behind[2],
                       object_right_behind[0], object_right_behind[1], object_right_behind[2]])

        if observation[1] <= 4 and object_front[3]:
            done = 1
            collision = 1
        if observation[4] >= -4 and object_behind[3]:
            done = 1
            collision = 1
        if observation[7] <= 4 and observation[8] <= 2 and object_left_front[3]:
            done = 1
            collision = 1
        if observation[10] <= 4 and observation[11] >= -2 and object_right_front[3]:
            done = 1
            collision = 1
        if observation[13] >= -4 and observation[14] <= 2 and object_left_behind[3]:
            done = 1
            collision = 1
        if observation[16] >= -4 and observation[17] >= -2 and object_right_behind[3]:
            done = 1
            collision = 1

        self.time += 1

        if self.time == 109:
            done = 1

        return observation, np.array(get_reward(observation, done, collision, action[0] + 10, reach)), np.array(done), collision

    def reset(self):
        self.scenario = self.scenarios[self.index_of_scenario]
        self.index_of_scenario += 1
        if self.index_of_scenario > len(self.scenarios) - 1:
            self.index_of_scenario = 0
        self.ego_track = self.scenario['EGO']
        self.object_tracks = []
        for key in self.scenario.keys():
            if key != 'EGO' and key != 'states':
                self.object_tracks.append(self.scenario[key])

        self.ego_x = self.ego_track.object_states[0].position[0]
        self.ego_y = self.ego_track.object_states[0].position[1]
        self.ego_yaw = self.ego_track.object_states[0].heading
        self.ego_v = sqrt(self.ego_track.object_states[0].velocity[0] ** 2 +
                          self.ego_track.object_states[0].velocity[1] ** 2)
        self.s = 0
        self.trajectory = []
        s = 0
        for i in range(len(self.ego_track.object_states)):
            if i == 0:
                self.trajectory.append(
                    [self.ego_track.object_states[i].position[0], self.ego_track.object_states[i].position[1],
                     s, self.ego_track.object_states[i].heading])
            else:
                delta_s = sqrt((self.ego_track.object_states[i].position[0] -
                                self.ego_track.object_states[i - 1].position[0]) ** 2 +
                               (self.ego_track.object_states[i].position[1] -
                                self.ego_track.object_states[i - 1].position[1]) ** 2)
                s += delta_s
                self.trajectory.append(
                    [self.ego_track.object_states[i].position[0], self.ego_track.object_states[i].position[1],
                     s, self.ego_track.object_states[i].heading])

        state = self.scenario['states'][0]
        ego_v = math.sqrt(state[0][3] ** 2 + state[0][4] ** 2)
        object_front_v = math.sqrt(state[1][5] ** 2 + state[1][6] ** 2)
        object_behind_v = math.sqrt(state[2][5] ** 2 + state[2][6] ** 2)
        object_left_front_v = math.sqrt(state[3][5] ** 2 + state[3][6] ** 2)
        object_right_front_v = math.sqrt(state[4][5] ** 2 + state[4][6] ** 2)
        object_left_behind_v = math.sqrt(state[5][5] ** 2 + state[5][6] ** 2)
        object_right_behind_v = math.sqrt(state[6][5] ** 2 + state[6][6] ** 2)
        observation = np.array([ego_v, state[1][0], state[1][1], object_front_v,
                       state[2][0], state[2][1], object_behind_v,
                       state[3][0], state[3][1], object_left_front_v,
                       state[4][0], state[4][1], object_right_front_v,
                       state[5][0], state[5][1], object_left_behind_v,
                       state[6][0], state[6][1], object_right_behind_v])
        self.time = 0
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(3,))
        self.viewer = None
        self.steps_beyond_done = None
        return observation

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        ego_x = 300
        ego_y = 200
        car_width = 20
        car_length = 40

        # number_of_car = len(self.object_position_for_view)
        from gym.envs.classic_control import rendering
        l, r, t, b = -car_width / 2, car_width / 2, car_length / 2, -car_length / 2
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            ego_car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            ego_car.add_attr(self.carttrans)
            ego_car.set_color(1, 0, 0)
            self.viewer.add_geom(ego_car)
            for i in range(len(self.object_tracks)):
                globals()['object_' + str(i)] = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                globals()['object_trans' + str(i)] = rendering.Transform()
                globals()['object_' + str(i)].add_attr(globals()['object_trans' + str(i)])
                self.viewer.add_geom(globals()['object_' + str(i)])

        # Edit the pole polygon vertex

        self.carttrans.set_translation(ego_x, ego_y)
        self.carttrans.set_rotation(self.ego_yaw - math.pi / 2)
        for i in range(len(self.object_tracks)):
            if self.object_tracks[i].object_states[0].timestep <= self.time <= self.object_tracks[i].object_states[
                -1].timestep:
                for object_state in self.object_tracks[i].object_states:
                    if object_state.timestep == self.time:
                        object_x_to_ego = ego_x + (object_state.position[0] - self.ego_x) * 10
                        object_y_to_ego = ego_y + (object_state.position[1] - self.ego_y) * 10
                        globals()['object_trans' + str(i)].set_translation(object_x_to_ego, object_y_to_ego)
                        globals()['object_trans' + str(i)].set_rotation(object_state.heading - math.pi / 2)
                        break
            else:
                globals()['object_trans' + str(i)].set_translation(10000, 10000)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
