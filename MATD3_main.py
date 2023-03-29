import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from replay_buffer import ReplayBuffer
from matd3 import MATD3
import copy
import gym
import gym_env
import time


class Runner:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Create env
        self.writer = SummaryWriter("runs/" + "MATD3_offline")

        # 生成环境
        self.env = gym.make('carla-v1')

        # self.env = make_env(env_name, discrete=False)  # Continuous action space
        # self.env_evaluate = make_env(env_name, discrete=False)

        # init_dict = {'gamma': 0.95, 'tau': self.args.tau, 'lr': self.args.lr,
        #              'hidden_dim': self.args.hidden_dim,
        #              'alg_types': ['MADDPG', 'MADDPG', 'MADDPG'],
        #              'agent_init_params': agent_init_params,
        #              'discrete_action': False}

        self.args.N = 1  # The number of agents
        self.args.obs_dim_n = [19]  # obs dimensions of N agents
        self.args.action_dim_n = [1]  # actions dimensions of N agents

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        print("Algorithm: MATD3")
        self.agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]

        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/{}/{}_env_{}_number_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_episode_steps = 0

        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self, ):
        # self.evaluate_policy()
        collision = 0
        success = 0
        frame = 0

        while self.total_episode_steps < self.args.max_episode_steps:
            obs_n = [self.env.reset()]
            print("Episodes %i-%i of %i" % (self.total_episode_steps + 1,
                                            self.total_episode_steps + 1 + 1,
                                            self.args.max_episode_steps))
            for n in range(self.args.episode_limit):
                # Each agent selects actions based on its own local observations(add noise for exploration)
                a_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]
                # --------------------------!!!注意！！！这里一定要deepcopy，MPE环境会把a_n乘5-------------------------------------------
                obs_next_n, r_n, done_n, collision = self.env.step(copy.deepcopy(a_n))
                if self.total_episode_steps % 100 == 0:
                    self.env.render()
                    time.sleep(0.1)

                # Store the transition
                
                self.replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done_n)
                obs_n = [obs_next_n]
                frame += 1

                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if self.replay_buffer.current_size > self.args.batch_size and frame % 100 < 1:
                    # Train each agent individually
                    print("start training!!!")
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

                # if self.total_episode_steps % self.args.evaluate_freq == 0:
                #     self.evaluate_policy()

                if done_n and collision != 1:
                    print('success!!!')
                    success += 1
                    break
                if collision == 1:
                    collision += 1
                    break
            self.env.close()
            print("*******************************************************************************************")
            print(r_n)
            print("*******************************************************************************************")
            if self.total_episode_steps > 20:
                self.writer.add_scalar("reward", r_n, frame)

            if self.total_episode_steps % 100 == 0:
                self.writer.add_scalar("Suceess rate", success / 100, frame)
                self.writer.add_scalar("Collision Rate", collision / 100, frame)
                success = 0
                collision = 0

            self.total_episode_steps += 1

        self.env.close()
        # self.env_evaluate.close()

    # def evaluate_policy(self, ):
    #     evaluate_reward = 0
    #     for _ in range(self.args.evaluate_times):
    #         obs_n = self.env_evaluate.reset()
    #         episode_reward = 0
    #         for _ in range(self.args.episode_limit):
    #             a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]  # We do not add noise when evaluating
    #             obs_next_n, r_n, done_n, _ = self.env_evaluate.step(copy.deepcopy(a_n))
    #             episode_reward += r_n[0]
    #             obs_n = obs_next_n
    #             if all(done_n):
    #                 break
    #         evaluate_reward += episode_reward
    #
    #     evaluate_reward = evaluate_reward / self.args.evaluate_times
    #     self.evaluate_rewards.append(evaluate_reward)
    #     print("total_steps:{} \t evaluate_reward:{} \t noise_std:{}".format(self.total_steps, evaluate_reward, self.noise_std))
    #     self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
    #     # Save the rewards and models
    #     np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.args.algorithm, self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
    #     for agent_id in range(self.args.N):
    #         self.agent_n[agent_id].save_model(self.env_name, self.args.algorithm, self.number, self.total_steps, agent_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_episode_steps", type=int, default=int(1e4), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    env_names = ["simple_speaker_listener", "simple_spread"]
    env_index = 0
    runner = Runner(args, env_name=env_names[env_index], number=1, seed=0)
    runner.run()
