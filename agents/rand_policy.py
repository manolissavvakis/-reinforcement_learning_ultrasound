from envs.us_env import PhantomUsEnv, random_env_generator
from envs.imaging import Probe, ImagingSystem
from envs.phantom import Teddy, ScatterersPhantom
from envs.generator import ConstProbeGenerator, ConstPhantomGenerator
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Generates given number of cyst phantom RF examples.""")

    parser.add_argument("--episodes", dest="no_episodes", type=int,
                        help="Number of episodes to generate.",
                        required=True)
    parser.add_argument("--steps", dest="no_steps", type=int,
                        help="Max episode length.",
                        required=True)
    parser.add_argument("--workers", dest="no_workers", type=int,
                        help="Number of MATLAB workers.",
                        required=True)
    parser.add_argument("--output_dir", dest="output_dir",
                        help="Destination directory for observations and env. vis.",
                        required=True)

    args = parser.parse_args()

    imaging = ImagingSystem(
        c=1540,
        fs=100e6,
        image_width=40/1000,
        image_height=90/1000,
        image_resolution=(40, 90),
        # image_grid=(40/1000, 90/1000),
        median_filter_size=5,
        dr_threshold=-100,
        #grid_step=0.5/1000
        no_lines=64,
    )

    # --- There were no phantom and probe generators, so i created them ---
    phantom = ScatterersPhantom(
        objects=[
            Teddy(
                belly_pos=np.array([0 / 1000, 0, 50 / 1000]), # X, Y, Z
                scale=12 / 1000,
                head_offset=.9
            )
        ],
        x_border=(-40 / 1000, 40 / 1000),
        y_border=(-40 / 1000, 40 / 1000),
        z_border=(0, 90 / 1000),
        n_scatterers=int(1e4),
        n_bck_scatterers=int(1e3),
        seed=42,
    )
    phantom_generator = ConstPhantomGenerator(phantom)

    probe = Probe(
        pos=np.array([0 / 1000, 0 / 1000, 0]),
        angle=0,
        width=40 / 1000,
        height=10 / 1000,
        focal_depth=30 / 1000
    )
    probe_generator = ConstProbeGenerator(probe)

    env = PhantomUsEnv(
        imaging=imaging,
        #env_generator=random_env_generator(),
        phantom_generator = phantom_generator,
        probe_generator = probe_generator,
        max_steps=args.no_steps,
        no_workers=args.no_workers
    )

    def plot_obj(d, i, ob, title):
        fig = plt.figure()
        plt.title(title)
        plt.imshow(ob, cmap='gray')
        plt.savefig(os.path.join(d, "step_%03d.png" % i))

    def plot_env(d, i):
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111, projection='3d')
        env.render_pyplot(ax)
        plt.savefig(os.path.join(d, "env_%03d.png" % i))


    for episode in range(args.no_episodes):
        print("Episode %d" % episode)
        env.reset()
        step = 0
        episode_dir = os.path.join(args.output_dir, "episode_%d" % episode)
        # just to obtain initial observation
        # ---- changed to 0 from [0, 0, 0]
        ob, reward, episode_over, _ = env.step(0)
        if args.output_dir:
            os.makedirs(episode_dir, exist_ok=True)
            plot_env(episode_dir, step)
            plot_obj(episode_dir, step, ob, "ep: %d, step: %d, reward: %s" % (episode, step, str(reward)))
        while not episode_over:
            step += 1
            print("Step %d" % step)
            action = env.action_space.sample()
            print("Performing action: %s" % str(action))
            start = time.time()
            ob, reward, episode_over, _ = env.step(action)
            end = time.time()
            print("Environment execution time: %d [s]" % (end-start))
            print("reward %f" % reward)
            with open(os.path.join(episode_dir, "log.txt"), 'a') as f:
                f.write("Episode %d, step %d\n" % (episode, step))
                f.write("Take action: %s\n" % action)
                f.write("Reward: %f\n" % reward)
                f.write(env.to_string())
            print(env.to_string())
            if args.output_dir:
                plot_obj(episode_dir, step, ob, "ep: %d, step: %d, reward: %s" % (episode, step, str(reward)))
                plot_env(episode_dir, step)
    print("Training is over (achieved max. number of episodes).")


