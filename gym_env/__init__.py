from gym.envs.registration import register

register(
    id='carla-v1',
    entry_point='gym_env.envs:OfflineRL',
)

register(
    id='carla-v2',
    entry_point='gym_env.envs:OfflineRL_test',
)