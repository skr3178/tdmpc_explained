#Reference: https://www.geeksforgeeks.org/understanding-prioritized-experience-replay/#

import gym
from gym.wrappers.monitoring import video_recorder
import numpy as np
import os
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.episode_boundaries = []
        self.current_episode = []

    def add(self, experience):
        self.current_episode.append(experience)
        if experience[4]:  # if done flag is True
            self.episode_boundaries.append((len(self.buffer), len(self.buffer) + len(self.current_episode)))
            self.buffer.extend(self.current_episode)
            self.current_episode = []

    def get_top_td_episodes(self, k=3):
        episode_scores = []
        for start, end in self.episode_boundaries:
            avg_td = np.mean([abs(x[5]) for x in list(self.buffer)[start:end]])
            episode_scores.append((avg_td, start, end))

        episode_scores.sort(reverse=True, key=lambda x: x[0])
        return [list(self.buffer)[start:end] for (_, start, end) in episode_scores[:k]]


def record_episode(env, episode, video_path):
    vid = video_recorder.VideoRecorder(env, path=video_path)
    env.reset()

    # CartPole specific state restoration
    initial_state = episode[0][0]
    env.env.state = initial_state  # Direct state manipulation

    vid.capture_frame()
    for transition in episode:
        _, action, _, _, _, _ = transition
        env.step(action)
        vid.capture_frame()
    vid.close()


if __name__ == "__main__":
    # Setup environment with older gym
    env = gym.make('CartPole-v1')
    buffer = ReplayBuffer(10000)

    # Collect episodes
    for _ in range(50):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            td_error = np.random.uniform(0, 2)  # Replace with real TD error
            buffer.add([state, action, reward, next_state, done, td_error])
            state = next_state

    # Record top episodes
    os.makedirs("td_error_videos", exist_ok=True)
    for i, episode in enumerate(buffer.get_top_td_episodes(k=3)):
        print(f"Recording episode {i} with avg TD error: {np.mean([x[5] for x in episode]):.2f}")
        video_path = f"td_error_videos/high_td_episode_{i}.mp4"
        record_episode(gym.make('CartPole-v1'), episode, video_path)

