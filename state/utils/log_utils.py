import csv
import os
import pathlib
import subprocess
import tempfile
from datetime import datetime

import absl.flags as flags
import cv2

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
import matplotlib
import ml_collections
import numpy as np
import wandb
from PIL import Image, ImageEnhance

matplotlib.use("Agg")


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        filtered_row = {
            k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)
        }
        filtered_row["step"] = step
        if self.file is None:
            # A resumed run (auto_resume / the process supervisor requeue) must not truncate the
            # log written before the restart. Preserve old column order while
            # allowing later phases and resumed processes to add metrics.
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                with open(self.path, newline="") as existing:
                    self.header = next(csv.reader(existing))
                self.file = open(self.path, "a", newline="")
            else:
                self.header = list(filtered_row)
                self.file = open(self.path, "w", newline="")
                csv.DictWriter(self.file, fieldnames=self.header).writeheader()

        new_columns = [key for key in filtered_row if key not in self.header]
        if new_columns:
            self._expand_header(self.header + new_columns)
        csv.DictWriter(self.file, fieldnames=self.header).writerow(filtered_row)
        self.file.flush()

    def _expand_header(self, header):
        """Atomically rewrite prior rows with blank cells for newly seen metrics."""
        self.file.close()
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                dir=os.path.dirname(os.path.abspath(self.path)),
                prefix=".csv-schema-",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                writer = csv.DictWriter(temporary, fieldnames=header)
                writer.writeheader()
                with open(self.path, newline="") as existing:
                    writer.writerows(csv.DictReader(existing))
                temporary.flush()
                os.fsync(temporary.fileno())
                os.chmod(temporary_path, os.stat(self.path).st_mode & 0o7777)
            os.replace(temporary_path, self.path)
            self.header = header
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)
            self.file = open(self.path, "a", newline="")

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ""
    exp_name += f"sd{seed:03d}_"
    if "the process supervisor_JOB_ID" in os.environ:
        exp_name += f"s_{os.environ['the process supervisor_JOB_ID']}."
    if "the process supervisor_PROCID" in os.environ:
        exp_name += f"{os.environ['the process supervisor_PROCID']}."
    exp_name += f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if "." not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project="project",
    group=None,
    name=None,
    mode="online",
    hyperparam_dict=None,
    log_code=True,
    tags=None,
    resume_id=None,
    resume_mode=None,
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    if tags is None:
        tags = [group] if group is not None else None

    # Combine flag dict with hyperparameters (similar to utils/wandb.py)
    config_dict = get_flag_dict()
    if hyperparam_dict is not None:
        config_dict.update(hyperparam_dict)

    init_kwargs = dict(
        config=config_dict,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )
    if resume_id is not None:
        init_kwargs.update(dict(id=resume_id, resume=resume_mode or "allow"))

    run = wandb.init(**init_kwargs)

    if log_code:
        run.log_code(
            root=".",
            include_fn=lambda p: (
                not any(
                    part in {".git", "__pycache__", "exp", ".onager"}
                    or part.startswith(".venv")
                    for part in pathlib.Path(p).parts
                )
            ),
        )

        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            diff = subprocess.check_output(["git", "diff"], text=True)
        except Exception:
            sha, diff = "no-git", ""
        wandb.config.update(
            {"git_commit": sha, "git_diff": diff}, allow_val_change=True
        )

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        padding = np.zeros((len_addition, t, h, w, c), dtype=v.dtype)
        v = np.concatenate((v, padding), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, (n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, (t, c, n_rows * h, n_cols * w))

    return v


def _figure_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def _resize_to_width(image, width):
    if image.shape[1] == width:
        return image
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _make_diagnostic_panel(diagnostics, width, marker_timestep=None):
    """Render Q and realized return-to-go below one episode video frame."""
    if not diagnostics or "q" not in diagnostics:
        return None

    import matplotlib.pyplot as plt

    q = np.asarray(diagnostics["q"]).reshape(-1)
    q_timesteps = np.asarray(diagnostics.get("q_timesteps", np.arange(len(q)))).reshape(
        -1
    )
    if q.size == 0:
        return None
    return_to_go = np.asarray(
        diagnostics.get("diagnostic_return_to_go", diagnostics.get("return_to_go", []))
    ).reshape(-1)
    return_timesteps = np.arange(len(return_to_go))
    episode_length = max(len(return_to_go), int(q_timesteps[-1]) + 1)
    # A chunk-start Q was predicted only at that state. Leave all intervening
    # timesteps empty rather than implying additional predictions/interpolation.
    q_by_timestep = np.full(episode_length, np.nan)
    q_by_timestep[q_timesteps] = q
    prediction_timesteps = np.arange(episode_length)
    return_label = (
        "realized return (critic units)"
        if "diagnostic_return_to_go" in diagnostics
        else "return-to-go"
    )
    has_bins = (
        "q_bin_probs" in diagnostics
        and np.asarray(diagnostics["q_bin_probs"]).ndim == 2
    )

    dpi = 100
    fig_width = max(width / dpi, 2.0)
    fig_height = max((0.48 if has_bins else 0.28) * fig_width, 1.4)

    if has_bins:
        probs = np.asarray(diagnostics["q_bin_probs"])
        fig, (line_ax, heat_ax) = plt.subplots(
            2,
            1,
            figsize=(fig_width, fig_height),
            dpi=dpi,
            height_ratios=(1, 3),
            sharex=True,
            constrained_layout=True,
        )
        line_ax.plot(
            prediction_timesteps,
            q_by_timestep,
            color="black",
            linewidth=1.3,
            marker=".",
            markersize=3,
            label="Q prediction",
        )
        if return_to_go.size:
            line_ax.plot(
                return_timesteps,
                return_to_go,
                color="tab:blue",
                linestyle="--",
                linewidth=1.3,
                label=return_label,
            )
        if marker_timestep is not None:
            line_ax.axvline(marker_timestep, color="red", linewidth=1.0)
        line_ax.set_ylabel("value", fontsize=7)
        line_ax.tick_params(labelsize=7, pad=1)
        line_ax.legend(fontsize=6, loc="best")

        probs_by_timestep = np.full((episode_length, probs.shape[-1]), np.nan)
        probs_by_timestep[q_timesteps] = probs
        image = heat_ax.imshow(
            np.ma.masked_invalid(probs_by_timestep.T),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(
                -0.5,
                episode_length - 0.5,
                -0.5,
                probs.shape[-1] - 0.5,
            ),
        )
        if marker_timestep is not None:
            heat_ax.axvline(marker_timestep, color="red", linewidth=1.0)
        heat_ax.set_xlabel("timestep", fontsize=7)
        heat_ax.set_ylabel("bin", fontsize=7)
        heat_ax.tick_params(labelsize=7, pad=1)
        cbar = fig.colorbar(image, ax=heat_ax, fraction=0.025, pad=0.015)
        cbar.ax.tick_params(labelsize=6, pad=1)
    else:
        fig, ax = plt.subplots(
            figsize=(fig_width, fig_height),
            dpi=dpi,
            constrained_layout=True,
        )
        ax.plot(
            prediction_timesteps,
            q_by_timestep,
            color="black",
            linewidth=1.4,
            marker=".",
            markersize=3,
            label="Q prediction",
        )
        if return_to_go.size:
            ax.plot(
                return_timesteps,
                return_to_go,
                color="tab:blue",
                linestyle="--",
                linewidth=1.4,
                label=return_label,
            )
        if marker_timestep is not None:
            ax.axvline(marker_timestep, color="red", linewidth=1.0)
        ax.set_xlabel("timestep", fontsize=7)
        ax.set_ylabel("value", fontsize=7)
        ax.tick_params(labelsize=7, pad=1)
        ax.legend(fontsize=6, loc="best")

    panel = _figure_to_rgb(fig)
    plt.close(fig)
    return _resize_to_width(panel, width).astype(np.uint8)


def _diagnostic_marker_timesteps(diagnostics, n_frames):
    q = np.asarray(diagnostics.get("q", []))
    if q.size == 0 or n_frames == 0:
        return [None] * n_frames

    render_timesteps = diagnostics.get("render_timesteps")
    if render_timesteps is not None:
        render_timesteps = np.asarray(render_timesteps).reshape(-1)
        if len(render_timesteps) == n_frames:
            episode_length = len(diagnostics.get("return_to_go", q))
            return [int(np.clip(t, 0, episode_length - 1)) for t in render_timesteps]

    q_timesteps = np.asarray(diagnostics.get("q_timesteps", np.arange(len(q)))).reshape(
        -1
    )
    return [
        int(round(t)) for t in np.linspace(q_timesteps[0], q_timesteps[-1], n_frames)
    ]


def append_diagnostics_to_renders(renders, video_diagnostics):
    """Stack each episode's q diagnostic panel below every frame."""
    if not video_diagnostics:
        return renders

    episode_panels = []
    panel_height = 0
    for idx, render in enumerate(renders):
        render = np.asarray(render)
        panels = []
        if idx < len(video_diagnostics):
            diagnostics = video_diagnostics[idx]
            if diagnostics and "q" in diagnostics:
                marker_timesteps = _diagnostic_marker_timesteps(
                    diagnostics,
                    len(render),
                )
                panels = [
                    _make_diagnostic_panel(
                        diagnostics,
                        render.shape[2],
                        marker_timestep=marker_timestep,
                    )
                    for marker_timestep in marker_timesteps
                ]
        episode_panels.append(panels)
        for panel in panels:
            if panel is not None:
                panel_height = max(panel_height, panel.shape[0])

    if panel_height == 0:
        return renders

    out = []
    for idx, render in enumerate(renders):
        render = np.asarray(render)
        panels = episode_panels[idx]
        if not panels:
            panels = [
                np.zeros((panel_height, render.shape[2], 3), dtype=np.uint8)
                for _ in range(len(render))
            ]
        padded_panels = []
        for panel in panels:
            if panel.shape[0] < panel_height:
                panel = np.pad(
                    panel,
                    ((0, panel_height - panel.shape[0]), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=255,
                )
            padded_panels.append(panel)
        panel_frames = np.stack(padded_panels, axis=0)
        out.append(np.concatenate([render, panel_frames], axis=1).astype(np.uint8))
    return out


def get_wandb_q_diagnostic_plots(video_diagnostics, prefix="evaluation"):
    """Create compact W&B line plots for the sampled video episodes."""
    plots = {}
    for episode_idx, diagnostics in enumerate(video_diagnostics):
        if "q" not in diagnostics or "return_to_go" not in diagnostics:
            continue
        q = np.asarray(diagnostics["q"]).reshape(-1)
        q_timesteps = np.asarray(
            diagnostics.get("q_timesteps", np.arange(len(q)))
        ).reshape(-1)
        return_to_go = np.asarray(
            diagnostics.get("diagnostic_return_to_go", diagnostics["return_to_go"])
        )[q_timesteps]
        plots[f"{prefix}/q_vs_return_to_go/episode_{episode_idx}"] = (
            wandb.plot.line_series(
                xs=q_timesteps,
                ys=[q, return_to_go],
                keys=["Q prediction", "realized return (critic units)"],
                title=f"Q accuracy: video episode {episode_idx}",
                xname="episode timestep",
            )
        )
    return plots


def get_wandb_video(
    renders=None,
    n_cols=None,
    fps=15,
    video_diagnostics=None,
    max_size=0,
):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
        video_diagnostics: Optional list of per-video diagnostic traces returned by rollout.
        max_size: Downsample the composed video's longest side to this size before
            MP4 encoding. Set to 0 to keep the original resolution.
    """
    renders = list(renders)
    if video_diagnostics is not None:
        renders = append_diagnostics_to_renders(renders, video_diagnostics)

    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(
            renders[i],
            ((0, 0), (1, 1), (1, 1), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    # Encode in-process to avoid forking FFmpeg after JAX has started threads.
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    frames = renders.transpose(0, 2, 3, 1)  # (t, h, w, c)
    if max_size and max(frames.shape[1:3]) > max_size:
        scale = max_size / max(frames.shape[1:3])
        output_width = int(round(frames.shape[2] * scale))
        output_width = max(2, output_width - output_width % 2)
        output_height = int(round(frames.shape[1] * scale))
        output_height = max(2, output_height - output_height % 2)
        frames = np.stack(
            [
                cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
                for frame in frames
            ]
        )
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        tmp.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {tmp.name}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    # Pass path to wandb.Video — it copies the file before this call returns,
    # so it is safe to leave cleanup to the OS (/tmp).
    return wandb.Video(tmp.name, format="mp4")
