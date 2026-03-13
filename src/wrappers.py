"""
wrappers.py — Gym environment wrappers for Super Mario Bros.

Applies frame skipping, grayscale, resize, and frame stacking.
"""

import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, RecordVideo


class SkipFrame(Wrapper):
    """Repeat the same action for `skip` frames and accumulate reward."""

    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info


def apply_wrappers(env, video_folder=None):
    """
    Apply preprocessing and optional video recording.
    """
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    
    if video_folder:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda x: x == 0, # Since we recreate env, it's always ep 0
            name_prefix="rl-video"
        )
        
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env
