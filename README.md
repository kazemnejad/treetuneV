# treetuneV: Suped-up Treetune!

- [Installation](#installation)
  - [Using `uv`](#using-uv)
  - [Using Singularity](#using-singularity)
- [Usage](#usage)
- [How To Use Launcher](#usage)
- [TreetuneV Specific Examples](#treetunev-specific-examples)
- [Trajectory Visualization](#trajectory-visualization)
- [FAQ](#faq)

## Installation

### Using `uv`

Read the setup guide: [`docs/setup_compute_canada.md`](docs/setup_compute_canada.md)

### Using Singularity

Download the singularity image from DockerHub and copy it to your scratch directory:

```bash
export SINGULARITY_CACHEDIR=~/scratch/.singularity_cache
singularity pull docker://kazemnejad/treetune_verl:v2
mkdir -p ~/scratch/containers
cp treetune_verl_v2.sif ~/scratch/containers/
```

**Launch an SSH server inside the singularity container to use VSCode/Cursor:**

First, add this to your `~/.ssh/config` (in your laptop), and replace `<MILA_USERNAME>` with your Mila username:

```bash
Host mila-container
  HostName localhost
  User <MILA_USERNAME>
  PreferredAuthentications publickey,keyboard-interactive
  Port 6322
  UserKnownHostsFile /dev/null
  StrictHostKeyChecking no
```

Then, run this inside the Mila cluster:

```bash
srun --partition=unkillable \
  --gres=gpu:1 -c 4 --mem 32G --time 10:00:00 \
  /path/to/treetune_verl/scripts/launch_ssh_inside_container.sh
```
Follow the on-screen instructions to forward the required ports, and finally, in VSCode/Cursor, use `mila-container` as the host for the remote SSH connection.

## Usage

```bash
export VERL_LOGGING_LEVEL=INFO

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export TREETUNEV__EXP_NAME=deepscaler_r1d-1.5b_grpo
export TREETUNEV__NNODE=1
export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS # should be set, otherwise defaults to 1

python -m verl.trainer.main_policy_iteration \
    --config-name="<config_name>"
```

Check out the examples in [`scripts/examples/`](scripts/examples/) for how to use VERL/TreetuneV.

## How To Use Launcher

Launcher syntax:
```bash
python scripts/launcher.py -i <image/venv-name> -s "<slurm-args>" <EXP_ID>
```
Optional arguments:
- `--interactive`: Run the experiment interactively.
- `--dry-run`: Do not submit the job to SLURM; Just create the SLURM script.
- `-n, --num_submissions`: Number submission. Submit the same experiment multiple times to run sequentially.
- `--dist_nodes`: Number of nodes for distributed training (defaults to 1).
### Compute Canada Examples
- **Nibi**: 


  `python scripts/launcher.py -i treetune_verl_v1.sh -s "-c 100 --mem 2000G --gpus=8 --time 25:00:00" <EXP_ID>`
- **Fir**: 

  `python scripts/launcher.py -i treetune_verl_v1.sh -s "-c 48 --mem 1000G --gpus=4 --time 25:00:00" <EXP_ID>`
- **Rorqual**: 

  `python scripts/launcher.py -i treetune_verl_v1.sh -s "-c 64 --mem 480G --gpus=4 --time 25:00:00" <EXP_ID>`
- **Tamia**: 

  `python scripts/launcher.py -i treetune_verl_v1.sh -s "-c 48 --mem 480G --gpus=4 --time 25:00:00" <EXP_ID>`

## TreetuneV Specific Examples

1. **GRPO using DeepScaleR recipe**: [`scripts/treetune/deepscaler_r1d-1.5b_grpo_stable.sh`](scripts/treetune/deepscaler_r1d-1.5b_grpo_stable.sh)


## Trajectory Visualization

Use the following command to visualize the flattened trajectories (general for all algorithms)

```bash
pip install textual==0.52.1
python scripts/rollout_viewer.py experiments/<exp_name>/train_rollouts
```

## FAQ

### How to setup the VSCode ray debugger?

Follow the guide: [VERL Ray Debug Tutorial](https://verl.readthedocs.io/en/latest/start/ray_debug_tutorial.html)

### Where is loss aggregation implemented?

Loss aggregation is implemented in [`verl/trainer/ppo/core_algos.py`](verl/trainer/ppo/core_algos.py) using the `agg_loss` and `agg_loss_with_trace_lengths` functions.

To change the aggregation mode, use `actor_rollout_ref.actor.loss_agg_mode` in the config.

### What does all the batch size configuration mean in the VERL universe?

- `data.train_batch_size` and `actor_rollout_ref.actor.ppo_mini_batch_size` are with respect to the number of prompts
- To map to TreeTune Next convention:
  - `number_of_episodes_per_iteration = data.train_batch_size * actor_rollout_ref.rollout.n`
  - `target_batch_size = actor_rollout_ref.actor.ppo_mini_batch_size * actor_rollout_ref.rollout.n`
- The number of gradient updates per iteration is `data.train_batch_size / actor_rollout_ref.actor.ppo_mini_batch_size`

### How to use smaller batch sizes for debugging/development?

Use configuration overrides like this:

```bash
python -m verl.trainer.main_policy_iteration \
    --config-name="<config_name>" \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.enable_debug=True \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    data.train_batch_size=4
```

### How can I add a new config?

Simply define it in YAML format. You can also assign a target config class to perform config validation. 

### Where are all important components in VERL implemented?

**Main Entry Point:**
- [`verl/trainer/main_policy_iteration.py`](verl/trainer/main_policy_iteration.py)

**Trainers:**
- [`verl/trainer/<algo_name>/ray_trainer.py`](verl/trainer/) - The main logic is inside the `trainer.fit()` method

**Actor (Trainer):**
- [`verl/workers/actor/dp_actor.py`](verl/workers/actor/dp_actor.py)

**Rollout (SGLang/Inference Engine):**
- [`verl/rollout/sglang_rollout/sglang_rollout.py`](verl/rollout/sglang_rollout/sglang_rollout.py)
- [`verl/rollout/sglang_rollout/custom_sglang_rollout.py`](verl/rollout/sglang_rollout/custom_sglang_rollout.py) - SGLang with custom modifications

**Dataset:**
- [`verl/utils/dataset/rl_dataset.py`](verl/utils/dataset/rl_dataset.py) - Dataset implementation
- [`verl/tasks/`](verl/tasks/) - Tasks that generate input data for `rl_dataset.py`

**Reward Function/Mechanism:**
- [`verl/utils/reward_score/treetune_math_verify.py`](verl/utils/reward_score/treetune_math_verify.py) - Actual reward function for math problems
- [`verl/workers/reward_manager/naive.py`](verl/workers/reward_manager/naive.py) - Naive reward manager (thin wrapper)

Reward managers wrap the reward function and handle some custom logics like length penalty, etc.

**ActorRolloutRefWorker:**
- [`verl/workers/fsdp_workers.py`](verl/workers/fsdp_workers.py)