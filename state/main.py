"""Train a standalone BC policy or fine-tune a policy with online RL."""

import gc
import os
import random
import signal
import sys
import time
import jax
import numpy as np
import tqdm
import wandb
from absl import app
from envs.benchmark import make_benchmark
from utils.evaluation import (
    EVAL_THROUGHPUT_FIXED_TIME_STEPS,
    add_eval_success_metrics,
    discounted_return,
    flatten,
    rollout,
)
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_wandb_q_diagnostic_plots, get_wandb_video
from utils.training_config import FLAGS
from utils.training_data import (
    prepare_training_data,
    prepare_dataset,
    prefill_replay_buffer,
    transform_replay_reward,
)
from utils.training_setup import prepare_run, create_agent, initialize_tracking

terminate_requested = False
_handler_called = False
_exception_occurred = False


def _sample_batch(dataset, batch_size, config, *, n_step=None):
    """Sample a training batch: chunk windows when action chunking or n-step
    returns are on (``Dataset.sample_sequence``), else single transitions."""
    n_step = FLAGS.n_step if n_step is None else n_step
    if FLAGS.horizon_length > 1 or n_step > 1:
        return dataset.sample_sequence(
            batch_size,
            sequence_length=FLAGS.horizon_length,
            discount=config.get("discount", 1.0),
            n_step=n_step,
            chunk_discount=FLAGS.chunk_discount,
        )
    return dataset.sample(batch_size)


def main(_):
    # Resolve the run directory and any checkpoint to resume.
    config = FLAGS.agent
    exp_name, resume_id = prepare_run(FLAGS, config)
    video_episodes = FLAGS.video_episodes if FLAGS.render_videos else 0
    bc_end = FLAGS.bc_steps
    offline_end = bc_end + FLAGS.offline_steps
    total_steps = offline_end + FLAGS.online_steps

    # Create benchmark resources and close them when training or evaluation exits.
    with make_benchmark(FLAGS, config) as benchmark:
        env, eval_env = benchmark.env, benchmark.eval_env
        venv = benchmark.make_vector_eval_env(FLAGS)
        _register_sigterm_handler()
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        if FLAGS.random_steps > 0:
            env.action_space.seed(FLAGS.seed)

        # Prepare datasets/replay, construct the agent, and restore saved state.
        (
            train_dataset,
            val_dataset,
            replay_buffer,
            example_batch,
            validation_discounted_return,
        ) = prepare_training_data(benchmark, FLAGS, config)
        agent = create_agent(FLAGS, config, example_batch)
        run = initialize_tracking(FLAGS, config, exp_name, resume_id)

        # A standalone evaluation restores a checkpoint and exits without training.
        if FLAGS.eval_only:
            assert FLAGS.restore_path, "--eval_only requires --restore_path"
            eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
            eval_info, trajs, cur_renders, video_diagnostics = rollout(
                agent=agent,
                env=eval_env,
                venv=venv,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=0.0,
                return_video_diagnostics=True,
                diagnostic_reward_scale=FLAGS.reward_scale,
                diagnostic_reward_bias=FLAGS.reward_bias,
                diagnostic_sparse=FLAGS.sparse,
                diagnostic_chunk_discount=FLAGS.chunk_discount,
                diagnostic_horizon_length=FLAGS.horizon_length,
            )
            eval_info = add_eval_success_metrics(eval_info, trajs)
            eval_metrics = {f"evaluation/{k}": v for k, v in eval_info.items()}
            wandb_metrics = dict(eval_metrics)
            if video_episodes > 0:
                wandb_metrics["video"] = get_wandb_video(
                    renders=cur_renders,
                    video_diagnostics=video_diagnostics,
                    fps=FLAGS.video_fps,
                    max_size=FLAGS.video_max_size,
                )
                wandb_metrics.update(get_wandb_q_diagnostic_plots(video_diagnostics))
            wandb.log(wandb_metrics, step=FLAGS.restore_epoch)
            eval_logger.log(eval_metrics, step=FLAGS.restore_epoch)
            eval_logger.close()
            wandb.finish()
            return

        # Initialize logging, episode statistics, action chunks, and update budget.
        train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
        eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
        first_time = time.time()
        last_time = time.time()
        start_step = FLAGS.restore_epoch
        i = start_step
        step = 0
        done = True
        pending_updates = 0.0
        expl_metrics = dict()
        online_rng = jax.random.PRNGKey(FLAGS.seed)
        action_queue = []
        action_dim = example_batch["actions"].shape[-1]
        episode_success = False
        online_episode_rewards = []
        online_episode_lengths = []
        online_episode_successes = []
        for i in tqdm.tqdm(
            range(start_step + 1, total_steps + 1),
            initial=start_step,
            total=total_steps,
            smoothing=0.1,
            dynamic_ncols=True,
        ):
            updated_this_step = False
            in_bc_phase = i <= bc_end
            in_offline_rl = bc_end < i <= offline_end
            in_online_phase = i > offline_end
            phase_step = (
                i if in_bc_phase else i - (bc_end if in_offline_rl else offline_end)
            )

            # Rotate offline shards only while the learner samples offline data.
            if (
                (not in_online_phase or FLAGS.balanced_sampling)
                and not FLAGS.override_dataset_path
                and benchmark.shards.should_rotate(i - 1)
            ):
                train_dataset = None
                gc.collect()
                train_dataset = prepare_dataset(benchmark.shards.load(i - 1), FLAGS)

            # BC updates only the BC policy; RL keeps that base policy frozen.
            if in_bc_phase:
                bc_agent = agent if config["agent_name"] == "bc" else agent.base_policy
                batch = _sample_batch(
                    train_dataset,
                    bc_agent.config["batch_size"],
                    bc_agent.config,
                    n_step=1,
                )
                bc_agent, update_info = bc_agent.update(batch)
                agent = (
                    bc_agent
                    if config["agent_name"] == "bc"
                    else agent.replace(base_policy=bc_agent)
                )
                updated_this_step = True
            elif in_offline_rl:
                # Offline RL performs Bellman/actor updates without collecting transitions.
                batch = _sample_batch(train_dataset, config["batch_size"], config)
                block = (phase_step - 1) // FLAGS.actor_update_block
                update_actor = (block + 1) % config.get("actor_update_period", 1) == 0
                agent, update_info = agent.update(batch, update_actor=update_actor)
                updated_this_step = True
            else:
                # Prefill once when entering online RL, after BC/offline RL completes.
                if phase_step == 1 and FLAGS.prefill_episodes > 0:
                    prefill_replay_buffer(
                        options=FLAGS,
                        agent=agent,
                        env=env,
                        replay_buffer=replay_buffer,
                        num_episodes=FLAGS.prefill_episodes,
                        cache_path=FLAGS.prefill_cache_path,
                        seed=FLAGS.seed,
                    )
                # Online RL executes queued action chunks, then stores one transition.
                online_rng, key = jax.random.split(online_rng)
                if done:
                    step = 0
                    replay_buffer.truncate_episode()
                    ob, _ = env.reset()
                    action_queue = []
                    online_episode_rewards = []
                if len(action_queue) == 0:
                    if phase_step <= FLAGS.random_steps:
                        action_queue.append(env.action_space.sample())
                    else:
                        sampled_action = agent.sample_actions(
                            observations=ob, temperature=1, seed=key
                        )
                        action_chunk = np.array(sampled_action).reshape(-1, action_dim)
                        action_queue.extend(action_chunk)
                action = action_queue.pop(0)
                next_ob, reward, terminated, truncated, info = env.step(action.copy())
                done = terminated or truncated
                episode_success = episode_success or np.any(
                    np.asarray(info.get("success", 0)) > 0
                )
                online_episode_rewards.append(np.asarray(reward).item())
                reward = transform_replay_reward(
                    reward,
                    sparse=FLAGS.sparse,
                    reward_scale=FLAGS.reward_scale,
                    reward_bias=FLAGS.reward_bias,
                )
                transition = dict(
                    observations=np.asarray(ob, dtype=np.float32),
                    actions=np.asarray(action, dtype=np.float32),
                    rewards=np.float32(reward),
                    terminals=np.float32(done),
                    masks=np.float32(
                        1.0 - (terminated if FLAGS.bootstrap_on_truncation else done)
                    ),
                    next_observations=np.asarray(next_ob, dtype=np.float32),
                )
                replay_buffer.add_transition(transition)
                ob = next_ob

                # Summarize completed episodes for the next training log entry.
                if done:
                    info_flat = flatten(info)
                    episode_length = info_flat.get("episode.length", step + 1)
                    episode_length = np.asarray(episode_length, dtype=np.float32).item()
                    online_episode_lengths.append(episode_length)
                    online_episode_successes.append(episode_success)
                    expl_metrics = {
                        f"env/{k}": np.mean(v) for k, v in info_flat.items()
                    }
                    expl_metrics["env/discounted_return"] = discounted_return(
                        online_episode_rewards, config.get("discount", 1.0)
                    )
                    recent_lengths = online_episode_lengths[-10:]
                    recent_successes = online_episode_successes[-10:]
                    success_rate = np.mean(recent_successes)
                    expl_metrics["env/throughput"] = (
                        EVAL_THROUGHPUT_FIXED_TIME_STEPS
                        / np.mean(recent_lengths)
                        * success_rate
                    )
                    if episode_success:
                        expl_metrics["env/success_episode.length"] = episode_length
                    episode_success = False
                step += 1

                # Accumulate fractional UTD and spend whole gradient updates.
                online_step = i - offline_end
                if online_step >= FLAGS.online_start_training:
                    pending_updates += FLAGS.utd
                while pending_updates >= 1:
                    pending_updates -= 1

                    # Mix offline/replay samples when required by the recipe.
                    half = config["batch_size"] // 2
                    if FLAGS.balanced_sampling:
                        dataset_batch = _sample_batch(train_dataset, half, config)
                        replay_batch = _sample_batch(replay_buffer, half, config)
                        batch = {
                            k: np.concatenate(
                                [dataset_batch[k], replay_batch[k]], axis=0
                            )
                            for k in dataset_batch
                        }
                    else:
                        batch = _sample_batch(
                            replay_buffer, config["batch_size"], config
                        )

                    # Critics learn on every update; actors follow their configured cadence.
                    block = (online_step - 1) // FLAGS.actor_update_block
                    update_actor = (block + 1) % config.get(
                        "actor_update_period", 1
                    ) == 0
                    agent, update_info = agent.update(batch, update_actor=update_actor)
                    updated_this_step = True

            # Log training losses, held-out losses, episode statistics, and timing.
            current_log_interval = (
                FLAGS.online_log_interval if in_online_phase else FLAGS.log_interval
            )
            should_log = updated_this_step and phase_step % current_log_interval == 0
            if should_log:
                train_metrics = {f"training/{k}": v for k, v in update_info.items()}
                train_metrics["training/rewards"] = batch["rewards"].mean()
                train_metrics["training/rewards_max"] = batch["rewards"].max()
                train_metrics["training/rewards_min"] = batch["rewards"].min()
                if val_dataset is not None:
                    val_batch = _sample_batch(val_dataset, config["batch_size"], config)
                    validation_agent = bc_agent if in_bc_phase else agent
                    _, val_info = validation_agent.total_loss(
                        val_batch, grad_params=None
                    )
                    train_metrics.update(
                        {f"validation/{k}": v for k, v in val_info.items()}
                    )
                    if validation_discounted_return is not None:
                        train_metrics["validation/discounted_return"] = (
                            validation_discounted_return
                        )
                train_metrics["time/epoch_time"] = (
                    time.time() - last_time
                ) / current_log_interval
                train_metrics["time/total_time"] = time.time() - first_time
                train_metrics.update(expl_metrics)
                last_time = time.time()
                wandb.log(train_metrics, step=i)
                train_logger.log(train_metrics, step=i)

            # Evaluate BC or the RL policy at the interval for its current phase.
            online_step = i - offline_end
            eval_interval = (
                FLAGS.online_eval_interval if in_online_phase else FLAGS.eval_interval
            )
            should_evaluate = eval_interval > 0 and phase_step % eval_interval == 0
            if should_evaluate:
                num_eval_episodes = FLAGS.eval_episodes
                if online_step > 0 and online_step % 1000000 == 0:
                    num_eval_episodes = FLAGS.million_step_eval_episodes
                renders = []
                eval_metrics = {}
                log_video_now = (
                    video_episodes > 0
                    and online_step > 0
                    and (
                        FLAGS.video_log_interval == 0
                        or online_step % FLAGS.video_log_interval == 0
                    )
                )
                eval_info, trajs, cur_renders, video_diagnostics = rollout(
                    agent=bc_agent if in_bc_phase else agent,
                    env=eval_env,
                    venv=venv,
                    num_eval_episodes=num_eval_episodes,
                    num_video_episodes=video_episodes if log_video_now else 0,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=0.0,
                    return_video_diagnostics=True,
                    diagnostic_reward_scale=FLAGS.reward_scale,
                    diagnostic_reward_bias=FLAGS.reward_bias,
                    diagnostic_sparse=FLAGS.sparse,
                    diagnostic_chunk_discount=FLAGS.chunk_discount,
                    diagnostic_horizon_length=FLAGS.horizon_length,
                )
                eval_info = add_eval_success_metrics(eval_info, trajs)
                renders.extend(cur_renders)
                for k, v in eval_info.items():
                    eval_metrics[f"evaluation/{k}"] = v
                wandb_metrics = dict(eval_metrics)
                if log_video_now:
                    video = get_wandb_video(
                        renders=renders,
                        video_diagnostics=video_diagnostics,
                        fps=FLAGS.video_fps,
                        max_size=FLAGS.video_max_size,
                    )
                    wandb_metrics["video"] = video
                    wandb_metrics.update(
                        get_wandb_q_diagnostic_plots(video_diagnostics)
                    )
                wandb.log(wandb_metrics, step=i)
                eval_logger.log(eval_metrics, step=i)

            # Record diagnostic videos between full policy evaluations.
            if (
                video_episodes > 0
                and FLAGS.video_log_interval > 0
                and (online_step > 0)
                and (online_step % FLAGS.video_log_interval == 0)
                and (not should_evaluate)
            ):
                _, _, video_renders, video_diagnostics = rollout(
                    agent=agent,
                    env=eval_env,
                    venv=venv,
                    num_eval_episodes=0,
                    num_video_episodes=video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=0.0,
                    return_video_diagnostics=True,
                    diagnostic_reward_scale=FLAGS.reward_scale,
                    diagnostic_reward_bias=FLAGS.reward_bias,
                    diagnostic_sparse=FLAGS.sparse,
                    diagnostic_chunk_discount=FLAGS.chunk_discount,
                    diagnostic_horizon_length=FLAGS.horizon_length,
                )
                video_metrics = {
                    "video": get_wandb_video(
                        renders=video_renders,
                        video_diagnostics=video_diagnostics,
                        fps=FLAGS.video_fps,
                        max_size=FLAGS.video_max_size,
                    )
                }
                video_metrics.update(get_wandb_q_diagnostic_plots(video_diagnostics))
                wandb.log(video_metrics, step=i)

            # Save learner checkpoints and optional replay snapshots for resuming.
            save_interval = (
                FLAGS.online_save_interval if in_online_phase else FLAGS.save_interval
            )
            should_save = (
                save_interval > 0 and phase_step % save_interval == 0
            ) or i in (bc_end, offline_end)
            if should_save:
                save_agent(agent, FLAGS.save_dir, i)
                if in_online_phase and FLAGS.save_online_buffer:
                    replay_buffer.save(FLAGS.save_dir, i)

            # On termination, save recoverable state before leaving the benchmark context.
            global terminate_requested
            if terminate_requested:
                try:
                    save_agent(agent, FLAGS.save_dir, i)
                    if in_online_phase:
                        replay_buffer.save(FLAGS.save_dir, i)
                    wandb.log({"system/sigterm": 1}, step=i)
                    train_logger.close()
                    eval_logger.close()
                    wandb.finish()
                except Exception as e:
                    print(f"Error saving checkpoint after SIGTERM: {e}", flush=True)
                print("Checkpoint saved after SIGTERM. Exiting now.", flush=True)
                sys.exit(0)

        # Persist the final learner and mark successful completion.
        save_agent(agent, FLAGS.save_dir, i)
        if i > offline_end and FLAGS.save_online_buffer:
            replay_buffer.save(FLAGS.save_dir, i)
        train_logger.close()
        eval_logger.close()
        wandb.finish()
        with open(os.path.join(FLAGS.save_dir, "done.txt"), "w") as f:
            f.write(str(run.id))


def _register_sigterm_handler():
    """Install SIGTERM handler. Called from main() AFTER AsyncVectorEnv creation
    so that forked worker processes do not inherit it."""

    def _handle_sigterm(signum, frame):
        global _handler_called, terminate_requested, _exception_occurred
        if _handler_called or _exception_occurred:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os._exit(1)
        _handler_called = True
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        terminate_requested = True
        print("Received SIGTERM: will save checkpoint and exit safely...", flush=True)

    signal.signal(signal.SIGTERM, _handle_sigterm)


def main_with_exception_handling(argv):
    """Wrapper around main() that handles exceptions to ensure clean exit and resource return to the process supervisor."""
    global _handler_called, _exception_occurred, terminate_requested
    try:
        main(argv)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except (KeyboardInterrupt, SystemExit):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        raise
    except EOFError:
        pass
    except Exception as e:
        _exception_occurred = True
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print(f"Error during training: {e}", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    app.run(main_with_exception_handling)
