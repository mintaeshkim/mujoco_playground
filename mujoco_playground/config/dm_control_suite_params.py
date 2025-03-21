"""RL config for DM Control Suite."""

from ml_collections import config_dict

from mujoco_playground._src import dm_control_suite


def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=60_000_000,
      num_evals=10,
      reward_scaling=10.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=30,
      num_minibatches=32,
      num_updates_per_batch=16,
      discounting=0.995,
      learning_rate=1e-3,
      entropy_cost=1e-2,
      num_envs=2048,
      batch_size=1024,
  )

  if env_name.startswith("AcrobotSwingup"):
    rl_config.num_timesteps = 100_000_000
  if env_name == "BallInCup":
    rl_config.discounting = 0.95
  elif env_name.startswith("Swimmer"):
    rl_config.num_timesteps = 100_000_000
  elif env_name == "WalkerRun":
    rl_config.num_timesteps = 100_000_000
  elif env_name == "FingerSpin":
    rl_config.discounting = 0.95
  elif env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4
    rl_config.num_updates_per_batch = 4

  return rl_config


def brax_vision_ppo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax Vision PPO config for the given environment."""
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      madrona_backend=True,
      wrap_env=False,
      num_timesteps=1_000_000,
      num_evals=5,
      reward_scaling=0.1,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=10,
      num_minibatches=8,
      num_updates_per_batch=8,
      discounting=0.97,
      learning_rate=5e-4,
      entropy_cost=5e-3,
      num_envs=1024,
      num_eval_envs=1024,
      batch_size=256,
      max_grad_norm=1.0,
  )

  if env_name != "CartpoleBalance":
    raise NotImplementedError(f"Vision PPO params not tested for {env_name}")

  return rl_config


def brax_sac_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,
      learning_rate=1e-3,
      num_envs=128,
      batch_size=512,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192,
      network_factory=config_dict.create(
          q_network_layer_norm=True,
      ),
  )

  if env_name == "PendulumSwingUp":
    rl_config.action_repeat = 4

  if (
      env_name.startswith("Acrobot")
      or env_name.startswith("Swimmer")
      or env_name.startswith("Finger")
      or env_name.startswith("Hopper")
      or env_name
      in ("CheetahRun", "HumanoidWalk", "PendulumSwingUp", "WalkerRun")
  ):
    rl_config.num_timesteps = 10_000_000

  return rl_config
