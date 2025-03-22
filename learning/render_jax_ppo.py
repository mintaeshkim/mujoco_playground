# render_from_checkpoint.py

import os
import jax
import jax.numpy as jp
import mujoco
import mediapy as media
from orbax import checkpoint as ocp
from flax.training import orbax_utils
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params, locomotion_params, dm_control_suite_params
from brax.training.agents.ppo import networks as ppo_networks

# checkpoint path
ckpt_path = "logs/CartpoleBalance-20250320-204728/checkpoints/55050240"
env_name = "CartpoleBalance"

# env config
env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=env_cfg)

# restore params
orbax_checkpointer = ocp.PyTreeCheckpointer()
params = orbax_checkpointer.restore(ckpt_path)

# build inference function
ppo_params = locomotion_params.brax_ppo_config(env_name)
networks = ppo_networks.make_ppo_networks(env.observation_size, env.action_size)
inference_fn = networks.make_inference_fn(params, deterministic=True)
jit_inference_fn = jax.jit(inference_fn)

# reset env
reset_rng = jax.random.PRNGKey(123)
state = jax.jit(env.reset)(reset_rng)
rollout = [state]

# rollout
for _ in range(env_cfg.episode_length):
    act_rng, reset_rng = jax.random.split(reset_rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jax.jit(env.step)(state, ctrl)
    rollout.append(state)
    if state.done:
        break

# render
render_every = 2
fps = 1.0 / env.dt / render_every
traj = rollout[::render_every]

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

frames = env.render(traj, height=480, width=640, scene_option=scene_option)
media.write_video("rendered.mp4", frames, fps=fps)
print("Saved rollout video to rendered.mp4")
