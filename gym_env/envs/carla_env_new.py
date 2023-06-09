#!/usr/bin/env python

from __future__ import division

import copy
import math
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla
import random


# MADDPG环境代码。如何编写可查阅carla_readme.py
class CarlaEnvNew(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        # parameters

        self._Kcte = float(0.5)
        self._closest_distance = float(0)
        self.type = 'MADDPG'

        self.dt = params['dt']
        self.max_time_episode = params['max_time_episode']
        self.punish_time_episode = params['punish_time_episode']
        self.desired_speed = params['desired_speed']
        self.dests = [[290, -246.3, 0.275307], [258.5, -290, 0.275307], [258.5, -270, 0.275307]]

        # Connect to carla server and get world object
        print('connecting to Carla server...')
        client = carla.Client('localhost', params['port'])
        client.set_timeout(10.0)
        self.world = client.load_world(params['town'])
        print('Carla server connected!')
        self.map = self.world.get_map()

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # # Get spawn points
        # self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # self.walker_spawn_points = []

        # Create the ego vehicle blueprint
        self.ego_bp = random.choice(self.world.get_blueprint_library().filter("vehicle.lincoln*"))
        self.ego_bp.set_attribute('color', "255,0,0")
        self.surround_bp1 = random.choice(self.world.get_blueprint_library().filter("vehicle.lincoln*"))
        self.surround_bp1.set_attribute('color', "255,128,0")
        self.surround_bp2 = random.choice(self.world.get_blueprint_library().filter("vehicle.lincoln*"))
        self.surround_bp2.set_attribute('color', "255,128,0")

        # # collision sensor
        self.collision_sensor = self.world.get_blueprint_library().find('sensor.other.collision')

        self.sensor_ego = None
        self.sensor_surround1 = None
        self.sensor_surround2 = None

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # # Initialize the renderer
        # self._init_renderer()

        # # Get pixel grid points
        # if self.pixor:
        #   x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
        #   x, y = x.flatten(), y.flatten()
        #   self.pixel_grid = np.vstack((x, y)).T

    def reset(self):
        # 重置环境的函数
        # Clear sensor objects
        self.collision_ego = False
        self.collision_surround1 = False
        self.collision_surround2 = False

        if self.sensor_ego is not None and self.sensor_ego.is_listening:
            self.sensor_ego.stop()
            self.sensor_surround1.stop()
            self.sensor_surround2.stop()

        self.sensor_ego = None
        self.sensor_surround1 = None
        self.sensor_surround2 = None
        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision',
                                'vehicle.*'])

        # Disable sync mode
        self._set_synchronous_mode(False)
        spaw_points = self.world.get_map().get_spawn_points()

        # self.vehicle_spawn_points0 = carla.Transform(
        #     carla.Location(x=181.899918 + np.random.uniform(-10, 10), y=58.910496, z=0.275307),
        #     carla.Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000))
        self.vehicle_spawn_points0 = carla.Transform(
            carla.Location(x=255.3, y=-271.3, z=0.275307),
            carla.Rotation(pitch=0.000000, yaw=90, roll=0.000000))
        self.ego_x = 255.3
        self.ego_y = -271.3
        self.ego_v = 6
        self.ego_yaw = 1.5708
        self.index = 1
        waypoint_ego = self.map.get_waypoint(carla.Location(x=255.3, y=-271.3, z=0.275307))
        self.waypoints_path = []
        for i in range(1, 500):
            next_waypoint = waypoint_ego.next(i * 0.1)
            for w in next_waypoint:
                if w.transform.location.x >= waypoint_ego.transform.location.x - 0.045:
                    self.waypoints_path.append(w)

        surround1_y = random.uniform(-240, -220)
        surround2_y = random.uniform(surround1_y + 5, -200)
        surround1_v = random.uniform(2, 6)
        surround2_v = random.uniform(2, surround1_v)
        self.vehicle_spawn_points1 = carla.Transform(carla.Location(x=258.5, y=surround1_y, z=0.275307),
                                                     carla.Rotation(pitch=0.000000, yaw=270, roll=0.000000))
        self.vehicle_spawn_points2 = carla.Transform(carla.Location(x=258.5, y=surround2_y, z=0.275307),
                                                     carla.Rotation(pitch=0.000000, yaw=270, roll=0.000000))
        self.ego = self.world.spawn_actor(self.ego_bp, self.vehicle_spawn_points0)

        self.surround1 = self.world.spawn_actor(self.surround_bp1, self.vehicle_spawn_points1)
        self.surround1.set_autopilot(False)
        self.surround2 = self.world.spawn_actor(self.surround_bp2, self.vehicle_spawn_points2)
        self.surround2.set_autopilot(False)

        self.ego.set_target_velocity(carla.Vector3D(0, self.desired_speed, 0))
        self.surround1.set_target_velocity(carla.Vector3D(0, -surround1_v, 0))
        self.surround2.set_target_velocity(carla.Vector3D(0, -surround2_v, 0))

        self.ego_terminal = 0
        self.surround1_terminal = 0
        self.surround2_terminal = 0

        self.ego_done = 1
        self.surround1_done = 1
        self.surround2_done = 1

        self.last_ego_x = 255.3
        self.last_ego_y = -271.3
        self.last_surround1_x = 258.5
        self.last_surround1_y = surround1_y
        self.last_surround2_x = 258.5
        self.last_surround2_y = surround2_y

        transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.sensor_ego = self.world.spawn_actor(self.collision_sensor, transform, attach_to=self.ego)

        self.sensor_surround1 = self.world.spawn_actor(self.collision_sensor, transform, attach_to=self.surround1)
        self.sensor_surround2 = self.world.spawn_actor(self.collision_sensor, transform, attach_to=self.surround2)

        def callback_ego(event):
            self.collision_ego = True
            print('collision!!!!!!!')

        def callback_surround1(event):
            self.collision_surround1 = True
            print('collision!!!!!!!')

        def callback_surround2(event):
            self.collision_surround2 = True
            print('collision!!!!!!!')

        self.sensor_ego.listen(callback_ego)
        self.sensor_surround1.listen(callback_surround1)
        self.sensor_surround2.listen(callback_surround2)

        # self.ego.set_target_velocity(carla.Vector3D(-self.desired_speed, 0, 0))
        # self.surround.set_target_velocity(carla.Vector3D(-self.desired_speed, 0, 0))

        # # Add camera sensor
        # self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        # self.camera_sensor.listen(lambda data: get_camera_img(data))
        #
        # def get_camera_img(data):
        #     array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        #     image = np.reshape(array, (data.height, data.width, 4))
        #
        #     # Get the r channel
        #     sem = image[:, :, 2]
        #     # print(sem)
        #     m = len(sem[0, :])
        #     if self.location_flag == None:
        #         for i in range(len(sem[:, 0])):
        #             for j in range(int(m / 2)):
        #                 if sem[i][j + int(m / 2)] == 4:
        #                     self.location_flag = True

        # print(self.location_flag)

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        # Set ego information for render

        return self._get_obs()

    def step(self, actions):
        # Calculate acceleration and steering
        action_ego = actions[0][0]
        action_surround1 = actions[0][1]
        action_surround2 = actions[0][2]
        if action_ego > 0:
            throttle_ego = action_ego * 0.4
            brake_ego = 0
        else:
            brake_ego = -action_ego * 0.8
            throttle_ego = 0
        if action_surround1 > 0:
            throttle_surround1 = action_surround1 * 0.4
            brake_surround1 = 0
        else:
            brake_surround1 = -action_surround1 * 0.8
            throttle_surround1 = 0
        if action_surround2 > 0:
            throttle_surround2 = action_surround2 * 0.4
            brake_surround2 = 0
        else:
            brake_surround2 = -action_surround2 * 0.8
            throttle_surround2 = 0

        # 测试车辆转弯的部分
        # brake = 0
        # throttle = 0
        # if action == 0:
        #     brake = 0.5
        # elif action == 1:
        #     brake = 0.25
        # elif action == 2:
        #     throttle = 0.2
        # elif action == 3:
        #     throttle = 0.3
        # elif action == 4:
        #     throttle = 0.4
        # elif action == 5:
        #     throttle = 0.5
        # elif action == 6:
        #     throttle = 0.6
        # elif action == 7:
        #     throttle = 0.7
        # elif action == 8:
        #     throttle = 0.8
        # elif action == 9:
        #     throttle = 0.9
        # else:
        #     throttle = 0

        # 选择waypoints
        shortest_distance = 3
        for i in range(self.index, len(self.waypoints_path) - 2):
            distance = np.linalg.norm(np.array([
                self.waypoints_path[i].transform.location.x - self.ego_x,
                self.waypoints_path[i].transform.location.y - self.ego_y]))
            if distance < shortest_distance:
                shortest_distance = distance
                self.index = i + 1
        # print(self.index)
        # print(self.ego_v)
        waypoints = [[self.waypoints_path[self.index].transform.location.x,
                      self.waypoints_path[self.index].transform.location.y],
                     [self.waypoints_path[self.index + 1].transform.location.x,
                      self.waypoints_path[self.index + 1].transform.location.y]]

        ###########################################
        # waypoints可视化部分
        # if self.time_step == 0:
        #     waypoint_ego = self.map.get_waypoint(carla.Location(x=255.3, y=-271.3, z=0.275307))
        #     next_waypoint = waypoint_ego.next(2)[0]
        #     next_next_waypoint = next_waypoint.next(2)[0]
        #     waypoints = [[next_waypoint.transform.location.x, next_waypoint.transform.location.y],
        #                  [next_next_waypoint.transform.location.x, next_next_waypoint.transform.location.y]]
        # else:
        #     ego_transform = self.ego.get_transform()
        #     waypoint_ego = self.map.get_waypoint(ego_transform.location)
        #     next_waypoint = waypoint_ego
        #     next_next_waypoint = waypoint_ego
        #     if len(waypoint_ego.next(2)) != 1:
        #         for point in waypoint_ego.next(2):
        #             if point.transform.location.x - waypoint_ego.transform.location.x >= 0:
        #                 next_waypoint = point
        #                 break
        #     else:
        #         next_waypoint = waypoint_ego.next(2)[0]
        #     if len(next_waypoint.next(2)) != 1:
        #         for point in next_waypoint.next(2):
        #             if point.transform.location.x >= next_waypoint.transform.location.x:
        #                 next_next_waypoint = point
        #                 break
        #     else:
        #         next_next_waypoint = next_waypoint.next(2)[0]
        #     waypoints = [[next_waypoint.transform.location.x, next_waypoint.transform.location.y],
        #                  [next_next_waypoint.transform.location.x, next_next_waypoint.transform.location.y]]
        # self.world.debug.draw_string(next_waypoint.transform.location, 'O', draw_shadow=False,
        #                              color=carla.Color(r=255, g=0, b=0), life_time=120.0,
        #                              persistent_lines=True)
        # self.world.debug.draw_string(next_next_waypoint.transform.location, 'O', draw_shadow=False,
        #                              color=carla.Color(r=255, g=0, b=0), life_time=120.0,
        #                              persistent_lines=True)
        ####################################################################

        # for w in self.waypoints_path:
        #     self.world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
        #                                  color=carla.Color(r=255, g=0, b=0), life_time=120.0,
        #                                  persistent_lines=True)
        # self.world.debug.draw_string(self.waypoints_path[self.index].transform.location, 'O', draw_shadow=False,
        #                              color=carla.Color(r=255, g=0, b=0), life_time=120.0,
        #                              persistent_lines=True)
        # self.world.debug.draw_string(self.waypoints_path[self.index + 2].transform.location, 'O', draw_shadow=False,
        #                              color=carla.Color(r=255, g=0, b=0), life_time=120.0,
        #                              persistent_lines=True)

        # 计算转向角
        v1 = [waypoints[0][0] - self.ego_x, waypoints[0][1] - self.ego_y]
        v2 = [np.cos(self.ego_yaw), np.sin(self.ego_yaw)]
        heading_error = self.get_heading_error(waypoints, self.ego_yaw)
        cte_error = self.get_steering_direction(v1, v2) * self.get_cte_heading_error(self.ego_v)
        steer = heading_error + cte_error

        # 添加控制
        # 测试车辆转弯的部分
        # act_ego = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        # self.ego.apply_control(act_ego)

        # 控制部分
        if self.ego_terminal:
            throttle_ego = 0
            steer = 0
            brake_ego = 1
        act_ego = carla.VehicleControl(throttle=float(throttle_ego), steer=float(steer), brake=float(brake_ego))
        self.ego.apply_control(act_ego)
        if self.surround1_terminal:
            throttle_surround1 = 0
            brake_surround1 = 1
        act_surround1 = carla.VehicleControl(throttle=float(throttle_surround1), steer=float(0),
                                             brake=float(brake_surround1))
        self.surround1.apply_control(act_surround1)
        if self.surround2_terminal:
            throttle_surround2 = 0
            brake_surround2 = 1
        act_surround2 = carla.VehicleControl(throttle=float(throttle_surround2), steer=float(0),
                                             brake=float(brake_surround2))
        self.surround2.apply_control(act_surround2)

        while True:
            try:
                self.world.tick()
                break
            except:
                time.sleep(2)


        # Update timesteps and car position 更新数据
        self.time_step += 1
        self.total_step += 1
        self.info = None
        self.ego_x = self.ego.get_transform().location.x
        self.ego_y = self.ego.get_transform().location.y
        self.ego_yaw = self.ego.get_transform().rotation.yaw / 180 * math.pi
        self.ego_v = np.sqrt(self.ego.get_velocity().x ** 2 + self.ego.get_velocity().y ** 2)
        # print('ego_v')
        # print(self.ego_v)

        return self._get_obs(), self._get_reward(), self._terminal(), self.info

    # 转角控制的函数
    ######################################################3
    def get_heading_error(self, waypoints, current_yaw):
        waypoint_delta_x = waypoints[1][0] - self.ego_x if waypoints[1][0] - self.ego_x != 0 else 0.00001
        waypoint_delta_y = waypoints[1][1] - self.ego_y if waypoints[1][1] - self.ego_y != 0 else 0.00001
        waypoint_heading = np.arctan(waypoint_delta_y / waypoint_delta_x)
        heading_error_mod = divmod((waypoint_heading - current_yaw), np.pi)[1]
        if np.pi / 2 < heading_error_mod < np.pi:
            heading_error_mod -= np.pi
        return heading_error_mod

    def get_steering_direction(self, v1, v2):
        corss_prod = v1[0] * v2[1] - v1[1] * v2[0]
        if corss_prod >= 0:
            return -1
        return 1

    def get_cte_heading_error(self, v):
        proportional_cte_error = self._Kcte * self._closest_distance
        if v == 0:
            v = 0.0001
        cte_heading_error = np.arctan(proportional_cte_error / v)
        cte_heading_error_mod = divmod(cte_heading_error, np.pi)[1]
        if np.pi / 2 < cte_heading_error_mod < np.pi:
            cte_heading_error_mod -= np.pi
        return cte_heading_error_mod

    ##############################################

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
    """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _get_obs(self):
        """Get the observations."""
        # 取得需要输出的数据

        # State observation
        ego_trans = self.ego.get_transform()
        egovehicle_v = self.ego.get_velocity()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_v_x = egovehicle_v.x
        ego_v_y = egovehicle_v.y

        surround1_trans = self.surround1.get_transform()
        surround1_v = self.surround1.get_velocity()
        surround1_x = surround1_trans.location.x
        surround1_y = surround1_trans.location.y
        surround1_v_x = surround1_v.x
        surround1_v_y = surround1_v.y

        surround2_trans = self.surround2.get_transform()
        surround2_v = self.surround2.get_velocity()
        surround2_x = surround2_trans.location.x
        surround2_y = surround2_trans.location.y
        surround2_v_x = surround2_v.x
        surround2_v_y = surround2_v.y
        ego_v = math.sqrt(ego_v_x ** 2 + ego_v_y ** 2)
        surround1_v = math.sqrt(surround1_v_x ** 2 + surround1_v_y ** 2)
        surround2_v = math.sqrt(surround2_v_x ** 2 + surround2_v_y ** 2)

        obs = [[surround1_x - ego_x, surround1_y - ego_y, surround1_v_x - ego_v_x, surround1_v_y - ego_v_y,
                surround2_x - ego_x, surround2_y - ego_y, surround2_v_x - ego_v_x, surround2_v_y - ego_v_y,
                self.dests[0][0] - ego_x, self.dests[0][1] - ego_y, ego_v_x, ego_v_y
                ],
               [ego_x - surround1_x, ego_y - surround1_y, ego_v_x - surround1_v_x, ego_v_y - surround1_v_y,
                surround2_x - surround1_x, surround2_y - surround1_y, surround2_v_x - surround1_v_x, surround2_v_y - surround1_v_x,
                self.dests[1][0] - surround1_x, self.dests[1][1] - surround1_y, surround1_v_x, surround1_v_y
                ],
               [surround1_x - surround2_x, surround1_y - surround2_y, surround1_v,  # surround1_v_x, surround1_v_y,
                ego_x - surround2_x, ego_y - surround2_y, ego_v,  # ego_v_x, ego_v_y,
                self.dests[2][0] - surround2_x, self.dests[2][1] - surround2_y, surround2_v  # surround2_v_x, surround2_v_y
                ]
               ]

        # distance_ego_to_surround1 = np.sqrt((surround1_x - ego_x) ** 2 + (surround1_y - ego_y) ** 2)
        # distance_ego_to_surround2 = np.sqrt((surround2_x - ego_x) ** 2 + (surround2_y - ego_y) ** 2)
        # distance_surround1_to_surround2 = np.sqrt((surround2_x - surround1_x) ** 2 + (surround2_y - surround1_y) ** 2)
        # distance_ego_to_goal = np.sqrt((self.dests[0][0] - ego_x) ** 2 + (self.dests[0][1] - ego_y) ** 2)
        # distance_surround1_to_goal = np.sqrt((self.dests[1][0] - surround1_x) ** 2 + (self.dests[1][1] - surround1_y) ** 2)
        # distance_surround2_to_goal = np.sqrt((self.dests[2][0] - surround2_x) ** 2 + (self.dests[2][1] - surround2_y) ** 2)
        #
        # obs = [[distance_ego_to_surround1, surround1_v,  # surround1_v_x, surround1_v_y,
        #         distance_ego_to_surround2, surround2_v,  # surround2_v_x, surround2_v_y,
        #         distance_ego_to_goal, ego_v  # ego_v_x, ego_v_y
        #         ],
        #        [distance_ego_to_surround1, ego_v,  # ego_v_x, ego_v_y,
        #         distance_surround1_to_surround2, surround2_v,  # surround2_v_x, surround2_v_y,
        #         distance_surround1_to_goal, surround1_v
        #         # surround1_v_x, surround1_v_y
        #         ],
        #        [distance_surround1_to_surround2, surround1_v,  # surround1_v_x, surround1_v_y,
        #         distance_ego_to_surround2, ego_v,  # ego_v_x, ego_v_y,
        #         distance_surround2_to_goal, surround2_v
        #         # surround2_v_x, surround2_v_y
        #         ]
        #        ]

        # relative location

        return obs

    def _get_reward(self):
        # 取得奖励
        """Calculate the step reward."""
        # # reward for speed tracking
        v_ego = self.ego.get_velocity()
        speed_ego = np.sqrt(v_ego.x ** 2 + v_ego.y ** 2)
        # print('speed_ego')
        # print(speed_ego)
        r_speed_ego = speed_ego
        if speed_ego > self.desired_speed:
            r_speed_ego = speed_ego - (speed_ego - self.desired_speed) ** 2
        # print('r speed rgo')
        # print(r_speed_ego)

        v_surround1 = self.surround1.get_velocity()
        speed_surround1 = np.sqrt(v_surround1.x ** 2 + v_surround1.y ** 2)
        r_speed_surround1 = speed_surround1
        if speed_surround1 > self.desired_speed:
            r_speed_surround1 = speed_surround1 - (speed_surround1 - self.desired_speed) ** 2
        # print('r speed 1')
        # print(r_speed_surround1)

        v_surround2 = self.surround2.get_velocity()
        speed_surround2 = np.sqrt(v_surround2.x ** 2 + v_surround2.y ** 2)
        r_speed_surround2 = speed_surround2
        if speed_surround2 > self.desired_speed:
            r_speed_surround2 = speed_surround2 - (speed_surround2 - self.desired_speed) ** 2
        # print('r speed 2')
        # print(r_speed_surround2)

        # reward for collision
        r_collision_ego = 0
        r_collision_surround1 = 0
        r_collision_surround2 = 0
        # ego_x = self.ego.get_transform().location.x
        # ego_y = self.ego.get_transform().location.y

        r_time = 0
        if self.time_step >= self.max_time_episode:
            r_time = -1

        # cost for acceleration
        a = self.ego.get_acceleration()
        acc = np.sqrt(a.x ** 2 + a.y ** 2)
        r_acc = -abs(acc ** 2)

        a1 = self.surround1.get_acceleration()
        acc1 = np.sqrt(a1.x ** 2 + a1.y ** 2)
        r_acc1 = -abs(acc1 ** 2)

        a2 = self.surround2.get_acceleration()
        acc2 = np.sqrt(a2.x ** 2 + a2.y ** 2)
        r_acc2 = -abs(acc2 ** 2)

        ego_x = self.ego.get_transform().location.x
        ego_y = self.ego.get_transform().location.y

        surround1_x = self.surround1.get_transform().location.x
        surround1_y = self.surround1.get_transform().location.y

        surround2_x = self.surround2.get_transform().location.x
        surround2_y = self.surround2.get_transform().location.y

        if self.collision_ego:
            r_collision_ego = -1

        if self.collision_surround1:
            r_collision_surround1 = -1

        if self.collision_surround2:
            r_collision_surround2 = -1

        r_success_ego = 0
        r_success_surround1 = 0
        r_success_surround2 = 0
        r_success_all = 0
        r_time_ego = r_time
        r_time_surround1 = r_time
        r_time_surround2 = r_time
        distance_ego = np.sqrt((ego_x - self.dests[0][0]) ** 2 + (ego_y - self.dests[0][1]) ** 2)
        distance_surround1 = np.sqrt((surround1_x - self.dests[1][0]) ** 2 + (surround1_y - self.dests[1][1]) ** 2)
        distance_surround2 = np.sqrt((surround2_x - self.dests[2][0]) ** 2 + (surround2_y - self.dests[2][1]) ** 2)
        if distance_ego < 2:
            r_success_ego = 1
            r_time_ego = 0

        if distance_surround1 < 2:
            r_success_surround1 = 1
            r_time_surround1 = 0

        if distance_surround2 < 2:
            r_success_surround2 = 1
            r_time_surround2 = 0

        if r_success_ego and r_success_surround1 and r_success_surround2:
            r_success_all = 1

        r_distance_ego = np.sqrt((self.last_ego_x - self.dests[0][0]) ** 2 + (self.last_ego_y - self.dests[0][1]) ** 2) - distance_ego
        r_distance_surround1 = np.sqrt((self.last_surround1_x - self.dests[1][0]) ** 2 + (self.last_surround1_y - self.dests[1][1]) ** 2) - distance_surround1
        r_distance_surround2 = np.sqrt((self.last_surround2_x - self.dests[2][0]) ** 2 + (self.last_surround2_y - self.dests[2][1]) ** 2) - distance_surround2

        self.last_ego_x = ego_x
        self.last_ego_y = ego_y
        self.last_surround1_x = surround1_x
        self.last_surround1_y = surround1_y
        self.last_surround2_x = surround2_x
        self.last_surround2_y = surround2_y

        r_ego = 1000 * r_collision_ego + max(min(r_acc, 10), -10) + max(min(r_speed_ego, 10), -10) + \
                400 * r_success_ego * self.ego_done + 400 * r_success_all + r_time_ego * 200 - distance_ego
        r_surround1 = 1000 * r_collision_surround1 + max(min(r_acc1, 10), -10) + max(min(r_speed_surround1, 10), -10) + \
                      400 * r_success_surround1 * self.surround1_done + 400 * r_success_all + r_time_surround1 * 200 - distance_surround1
        r_surround2 = 1000 * r_collision_surround2 + max(min(r_acc2, 10), -10) + max(min(r_speed_surround2, 10), -10) + \
                      400 * r_success_surround2 * self.surround2_done + 400 * r_success_all + r_time_surround2 * 200 - distance_surround2


        if r_success_ego:
            self.ego_done = 0
        if r_success_surround1:
            self.surround1_done = 0
        if r_success_surround2:
            self.surround2_done = 0

        return [r_ego, r_surround1, r_surround2]

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # 判断状态，是否结束或是碰撞
        # # Get ego state
        # ego_x, ego_y = get_pos(self.ego)
        #
        # # If collides
        # if len(self.collision_hist)>0:
        #   return True

        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)

        ego_x = self.ego.get_transform().location.x
        ego_y = self.ego.get_transform().location.y

        surround1_x = self.surround1.get_transform().location.x
        surround1_y = self.surround1.get_transform().location.y

        surround2_x = self.surround2.get_transform().location.x
        surround2_y = self.surround2.get_transform().location.y

        # person_x = self.person.get_transform().location.x
        # person_y = self.person.get_transform().location.y
        # if abs(person_x - ego_x) < 3 and abs(person_y - ego_y) < 2.5:
        #     print(abs(person_x - ego_x), abs(person_y - ego_y))
        #     print("ego vehicle speed:", speed)
        #     return True
        if self.collision_ego or self.collision_surround1 or self.collision_surround2:
            print("collision!!!!!!")
            return [self.ego_terminal, self.surround1_terminal, self.surround2_terminal, True, False]

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            print("out of time!!!!!!")
            return [self.ego_terminal, self.surround1_terminal, self.surround2_terminal, False, True]

        # If at destination
        if np.sqrt((ego_x - self.dests[0][0]) ** 2 + (ego_y - self.dests[0][1]) ** 2) < 2:
            print('ego reach goal!!!!!')
            self.ego_terminal = 1

        if np.sqrt((surround1_x - self.dests[1][0]) ** 2 + (surround1_y - self.dests[1][1]) ** 2) < 2:
            print('surround1 reach goal!!!!!')
            self.surround1_terminal = 1

        if np.sqrt((surround2_x - self.dests[2][0]) ** 2 + (surround2_y - self.dests[2][1]) ** 2) < 2:
            print('surround2 reach goal!!!!!')
            self.surround2_terminal = 1

        return [self.ego_terminal, self.surround1_terminal, self.surround2_terminal, False, False]

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                # print(actor)
                if actor.is_alive:
                    # print('kill')
                    # print(actor)
                    actor.destroy()
