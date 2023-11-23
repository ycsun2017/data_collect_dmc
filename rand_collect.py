import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy

import utils
from logger import Logger
from video import VideoRecorder

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='log/', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    # new
    parser.add_argument('--load_model_from', default=None, type=str)
    parser.add_argument('--load_model_step', default=0, type=int)
    parser.add_argument('--load_buffer_from', default=None, type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat
    )
    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    
    print("work dir", args.work_dir, os.path.exists(args.work_dir))
    print("buffer dir", buffer_dir, os.path.exists(buffer_dir))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    env.reset()
    print("obs shape", env.observation_space.shape)
    print("physics shape", env.get_physics_shape())
    print("pixel shape", env.get_pixel_space().shape)

    replay_buffer = utils.FullReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        state_shape=env.get_physics_shape(),
        pixel_shape=env.get_pixel_space().shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_steps):
        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)
            
            if step % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # sample action for data collection
        action = env.action_space.sample()
        
        # obtain current physics state and pixel observation
        physics_state = env.get_physics_state()
        pixel_obs = env.get_pixel()

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool, physics_state, pixel_obs)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    main()
