import gym
import rlkit
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from lib import dqn_model, common

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='pong',
                        help='Environment name [pong, breakout]')
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='Enable CUDA')
    parser.add_argument('--save', '-s', default=None,
                        help='Saved file name, will be appened with -best.dat')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    params = common.HYPERPARAMS[args.env]

    env = gym.make(params['env_name'])
    env = rlkit.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment='-' + params['run_name'] + \
                '-basic')
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = rlkit.agent.TargetNet(net)
    selector = rlkit.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = rlkit.agent.DQNAgent(net, selector, device=device)

    exp_source = rlkit.experience.ExperienceSourceFirstLast(
            env,
            agent,
            gamma=params['gamma'],
            steps_count=1
    )
    exp_buffer = rlkit.experience.ExperienceReplayBuffer(
            exp_source,
            buffer_size=params['replay_size']
    )
    optimizer = optim.Adam(net.parameters(),
                           lr=params['learning_rate'])

    frame_idx = 0

    with common.RewardTracker(writer, 
                              params['stop_reward'], 
                              model=net if args.save is not None else None, 
                              save_name=args.save) as reward_tracker:
        while True:
            frame_idx += 1
            exp_buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0],
                                         frame_idx,
                                         selector.epsilon):
                    break
            
            if len(exp_buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = exp_buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch,
                                          net,
                                          tgt_net.target_model,
                                          gamma=params['gamma'],
                                          device=device)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
