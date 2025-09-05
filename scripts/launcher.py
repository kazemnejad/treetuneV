#!/usr/bin/env python3

import argparse
import copy
import datetime
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import Any, Optional, Union

import yaml


def create_md5_hash(inp: str):
    # Create MD5 hash object
    md5 = hashlib.md5()
    # Update the hash with the string
    md5.update(inp.encode("utf-8"))
    # Get the hexadecimal representation of the hash
    return md5.hexdigest()


def make_executable(script_path):
    mode = os.stat(str(script_path)).st_mode
    mode |= (mode & 0o444) >> 2
    os.chmod(str(script_path), mode)


def get_tempfile_path():
    return Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())


def get_repo_dir() -> Path:
    return Path(__file__).parent.parent


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None


def save_and_make_executable(job_path, script):
    with open(job_path, "w") as f:
        f.write(script)
    make_executable(job_path)


def replace_env_vars(target_str: str):
    for key, value in os.environ.items():
        target_str = target_str.replace(f"${key}", value)

    return target_str


def find_n_free_ports(
    n: int,
    seed: int = 42,
    max_attempts: int = 1000,
    generator: Optional[random.Random] = None,
    exclude: Optional[list[int]] = None,
) -> list[int]:
    ports = []
    attempts = 0

    if generator is None:
        generator = random.Random(seed)

    if exclude is None:
        exclude = []

    while len(ports) < n and attempts < max_attempts:
        port = generator.randint(1024, 65533)
        if port in ports or port in exclude:
            continue
        ports.append(port)
        attempts += 1

    if len(ports) < n:
        raise RuntimeError(f"Could not find {n} free ports after {max_attempts} attempts.")

    return ports


class ComputingCluster:
    def __init__(
        self,
        launcher_id: str,
        project_name: str = "treetune_verl",
        config: dict[str, str] = None,
        wandb_api_key: Optional[str] = None,
        wandb_project_name: Optional[str] = None,
        wandb_entity_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        dry_run: bool = False,
        interactive: bool = False,
        env_vars: list[str] = None,
        **kwargs,
    ):
        if config is None:
            config = {}

        self.config = config
        self.launcher_id = launcher_id
        self.project_name = self.config.get("wandb_project_name", project_name)
        self.interactive = interactive
        self.env_vars = env_vars
        # Priority order: 1. arg  2. env 3. config 4. default
        if wandb_api_key is None:
            wandb_api_key = os.environ.get("WANDB_API_KEY", self.config.get("wandb_api_key"))
        if wandb_project_name is None:
            wandb_project_name = os.environ.get("WANDB_PROJECT", self.config.get("wandb_project_name"))
        if wandb_entity_name is None:
            wandb_entity_name = os.environ.get("WANDB_ENTITY", self.config.get("wandb_entity_name"))
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN", self.config.get("hf_token"))

        self.wandb_api_key = wandb_api_key
        self.wandb_project_name = wandb_project_name
        self.wandb_entity_name = wandb_entity_name
        self.hf_token = hf_token

        # Make sure they are always set as env vars when running the job
        os.environ["WANDB_PROJECT"] = self.wandb_project_name
        os.environ["WANDB_ENTITY"] = self.wandb_entity_name
        os.environ["WANDB_API_KEY"] = self.wandb_api_key

        if self.hf_token is not None:
            os.environ["HF_TOKEN"] = self.hf_token

        self.dry_run = dry_run

        self.nnodes = int(kwargs.get("dist_nodes", 1))

    def setup_cluster(self) -> None:
        pass

    def prepare_job(self, output_dir: Path) -> str:
        pass

    def create_launch_script(self, job_body) -> Path:
        pass

    def execute_job(self, job_body):
        pass


FIND_FREE_PORT_BASH_FUNC = r"""
find_free_port() {
    local port
    while :; do
        port=$((RANDOM % 16384 + 49152))
        if ! lsof -i :"$port" >/dev/null 2>&1; then
            echo "$port"
            return
        fi
    done
}

"""

FIND_HEAD_NODE_IP_BASH_FUNC = r"""
find_head_node_ip() {
    local nodes
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    local nodes_array=($nodes)

    local head_node="${nodes_array[0]}"

    # Get the IP address of the head node
    local head_node_ip
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # If the IP contains spaces, it may be both IPv6 and IPv4; extract IPv4
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<< "$head_node_ip"
        if [[ ${#ADDR[0]} -gt 16 ]]; then
            head_node_ip="${ADDR[1]}"
        else
            head_node_ip="${ADDR[0]}"
        fi
        echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    echo "$head_node_ip"
}

"""


class SlurmComputingCluster(ComputingCluster):
    def __init__(
        self,
        slurm_args: str,
        images_dir: str = "containers",
        venvs_dir: str = "venvs",
        image_name: str = "latest.sif",
        num_submissions: int = 1,
        logs_dir: str = None,
        scripts_dir: str = None,
        shared_storage_dir: str = "$SCRATCH",
        compute_storage_dir: str = "$SLURM_TMPDIR",
        singularity_module: str = "singularity",
        runtime_env: str = "singularity",
        httpproxy_module: Optional[str] = None,
        github_token: str = None,
        wait_for_login_script: bool = False,
        wandb_offline: bool = False,
        hf_hub_offline: bool = False,
        account: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_submissions = num_submissions
        if num_submissions > 1 and self.interactive:
            raise ValueError("Cannot run multiple submissions in interactive mode")
        if logs_dir is None:
            logs_dir = f"~/scratch/{self.project_name}/sbatch_logs/"
        if scripts_dir is None:
            scripts_dir = f"~/scratch/{self.project_name}/jobs_scripts/"

        self.cluster_shared_storage_dir = Path(replace_env_vars(shared_storage_dir)).expanduser()
        self.compute_node_storage_dir = compute_storage_dir
        self.global_logs_dir = self.cluster_shared_storage_dir / self.project_name / "sbatch_logs"
        self.global_scripts_dir = self.cluster_shared_storage_dir / self.project_name / "jobs_scripts"

        self.singularity_image_library_path = self.cluster_shared_storage_dir / images_dir
        self.python_venv_library_path = self.cluster_shared_storage_dir / venvs_dir

        self.log_dir = Path(self.global_logs_dir) / f"lid_{self.launcher_id}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.script_dir = Path(self.global_scripts_dir)
        self.script_dir.mkdir(parents=True, exist_ok=True)

        self.runtime_env = runtime_env
        self.image_name = image_name
        self.slurm_args = slurm_args

        self.github_token = github_token
        self.wait_for_login_script = wait_for_login_script

        self.run_script_name = "worker_job.sh"

        self.wandb_offline = wandb_offline
        self.hf_hub_offline = hf_hub_offline
        self.account = account

        self.experiments_dir = self.cluster_shared_storage_dir / self.project_name / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.singularity_module = singularity_module
        self.singularity_cmd = "singularity"
        self.httpproxy_module = httpproxy_module

        if self.wandb_entity_name.endswith("-mila-org") and self.wandb_offline:
            self.wandb_offline = False
            self.httpproxy_module = "httpproxy"

    def _create_worker_script(self) -> str:
        """Create the worker script content with CUDA environment setup."""
        worker_script = f"#!/bin/bash\n\n"
        worker_script += "chmod a+x run.sh\n"
        worker_script += "./run.sh\n\n"
        return worker_script

    def prepare_job(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)

        import wandb

        # Priority order: 1. env 2. config 3. default
        project = self.wandb_project_name
        entity = self.wandb_entity_name
        api_key = self.wandb_api_key

        overrides = {}
        if project is not None:
            overrides["project"] = project
        if entity is not None:
            overrides["entity"] = entity
        api = wandb.Api(
            overrides=overrides if len(overrides) > 0 else None,
            api_key=api_key,
        )

        if entity is not None:
            artifact_name = f"{entity}/{project}/"
        else:
            artifact_name = ""

        artifact_name += f"bundle-{self.launcher_id}:latest"
        artifact = api.artifact(artifact_name)
        artifact.download(str(output_dir))

        try:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            persistent_key = metadata["exp_name"]
        except Exception as e:
            print("Unable to load metadata.json, computing persistent_dir based on launcher_id")
            persistent_key = create_md5_hash(self.launcher_id)

        worker_script = self._create_worker_script()
        save_and_make_executable(output_dir / self.run_script_name, worker_script)

        return persistent_key

    def execute_job(self, job_body):
        login_script_path, compute_script_path, persistent_key = self.create_launch_script(job_body)

        if self.dry_run:
            print("Dry run, not executing job")
            print(f"Login script: {login_script_path}")
            print(f"Compute script: {compute_script_path}")
            return

        if self.interactive:
            try:
                subprocess.check_call([login_script_path])
            except subprocess.CalledProcessError as e:
                print(e)
                print("Exiting...")
        else:
            print("Started executing...")
            print("To check all logs, visit this directory:")
            print(f"$ cd {self.log_dir} && ls -lh")

            log_path = self.log_dir / "launcher.txt"
            log_file = open(log_path, "w")
            p = subprocess.Popen(
                [login_script_path],
                start_new_session=True,
                stdout=log_file,
                stderr=log_file,
            )

            if self.wait_for_login_script:
                p.wait()
        return persistent_key

    def create_launch_script(self, job_body) -> tuple[Path, Path, str]:
        tmp_exp_dir = self._get_temp_exp_dir()

        persistent_key = self.prepare_job(tmp_exp_dir / "home")

        compute_script = self.create_compute_script(tmp_exp_dir, persistent_key)
        compute_script_path = self.script_dir / f"{self.launcher_id}_compute.sh"
        save_and_make_executable(compute_script_path, compute_script)

        login_script = self._create_pre_sbatch_launch_script(tmp_exp_dir, persistent_key)
        login_script += self._create_sbatch_launch_script(compute_script_path, persistent_key)
        login_script += self._create_post_sbatch_launch_script(tmp_exp_dir, persistent_key)

        login_script_path = self.script_dir / f"{self.launcher_id}_login.sh"
        save_and_make_executable(login_script_path, login_script)

        return login_script_path, compute_script_path, persistent_key

    def _get_temp_exp_dir(self):
        tmp_exp_dir = self.cluster_shared_storage_dir / self.project_name / "job_launcher_files" / f"{self.launcher_id}"
        tmp_exp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_exp_dir

    def _create_pre_sbatch_launch_script(self, tmp_exp_dir: Path, persistent_key: str) -> str:
        script = "#!/bin/bash \n\n"

        script += f"EXP_ROOT_DIR={self.experiments_dir}/{persistent_key}/ \n"
        script += f"LOG_DIR={self.log_dir}\n\n"
        script += "mkdir -p $EXP_ROOT_DIR\n"
        script += "ln -sfn $EXP_ROOT_DIR $LOG_DIR/exp_dir\n"
        script += "sleep 5\n\n"

        script += 'echo "Copying credentials to container..."\n'
        script += f"cp -r $HOME/.ssh {tmp_exp_dir}/home/\n\n"

        # Check if the job has `pre_submit_script.sh`
        pre_submit_script_path = tmp_exp_dir / "home" / "pre_submit_script.sh"
        if pre_submit_script_path.exists():
            # Prepare the environment for the pre_submit_script.sh
            # Export WANDB ENV variables
            if self.wandb_api_key is not None:
                script += f"export WANDB_API_KEY={self.wandb_api_key}\n"
            if self.wandb_project_name is not None:
                script += f"export WANDB_PROJECT={self.wandb_project_name}\n"
            if self.wandb_entity_name is not None:
                script += f"export WANDB_ENTITY={self.wandb_entity_name}\n"
            if self.hf_token is not None:
                script += f"export HF_TOKEN={self.hf_token}\n"

            script += f"export HF_HOME={self.experiments_dir}/hf_home\n"

            script += 'echo "Running pre_submit_script.sh..."\n\n'
            script += f"echo '{pre_submit_script_path}'\n"
            script += f"chmod a+x {pre_submit_script_path}\n"
            script += f"(cd {tmp_exp_dir}/home && ./{pre_submit_script_path.name})\n"

            # Clean up the environment for the pre_submit_script.sh
            script += "\nunset WANDB_API_KEY\n"
            script += "unset WANDB_PROJECT\n"
            script += "unset WANDB_ENTITY\n"
            script += "unset HF_TOKEN\n"
            script += "unset HF_HOME\n\n"

        return script

    def _slurm_safe_job_name(self, name: str) -> str:
        """Sanitize and bound job name for Slurm.

        - Keep alnum, dash, underscore; replace others with '-'
        - Collapse multiple dashes
        - Trim length to a safe bound (<= 128 chars)
        """
        # Replace invalid chars with '-'
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
        # Collapse duplicate '-'
        name = re.sub(r"-+", "-", name).strip("-")
        # Slurm allows up to 128 chars for job name; keep margin
        return name[:120] if len(name) > 120 else name

    def _create_sbatch_launch_script(self, compute_script_path: Path, persistent_key: str) -> str:
        script = 'echo "Submitting the job..." \n'
        script += f"cd {self.cluster_shared_storage_dir}\n"
        # Set job name as "<launcher_id>_<persistent_key>"
        persistent_part = self._slurm_safe_job_name(str(persistent_key))
        base = f"{self.launcher_id}_{persistent_part}" if persistent_part else f"{self.launcher_id}"
        job_name = self._slurm_safe_job_name(base)
        if not self.interactive:
            if self.num_submissions > 1:
                script += (
                    f"sbatch --array=1-{self.num_submissions}%1 "
                    f"--job-name={job_name} {self.slurm_args} {compute_script_path} \n"
                )
            else:
                script += f"sbatch --job-name={job_name} {self.slurm_args} {compute_script_path} \n"
        else:
            script += 'printf "\\n\\n--------------------------------------------------\\n"; \n'
            script += 'printf "Run the following command once the job is granted:\\n"; \n'
            script += f'echo "$ source {compute_script_path}";\n'
            script += 'echo "--------------------------------------------------"; \n'
            account_str = f"--account={self.account}" if self.account else ""
            script += f"salloc --job-name={job_name} {self.slurm_args} {account_str}\n"
        return script

    def _create_post_sbatch_launch_script(self, tmp_exp_dir: Path, persistent_key: str) -> str:
        script = ""
        return script

    def create_compute_script(self, tmp_exp_dir: Path, persistent_key: str) -> str:
        script = "#!/bin/bash\n"
        script += f"#SBATCH -o {self.log_dir}/compute-%j.out\n"
        script += f"#SBATCH --ntasks-per-node=1\n"
        script += f"#SBATCH --nodes={self.nnodes}\n"
        if self.account is not None:
            script += f"#SBATCH --account={self.account}\n"

        if self.nnodes > 1:
            script += FIND_FREE_PORT_BASH_FUNC
            script += FIND_HEAD_NODE_IP_BASH_FUNC

        script += "\n"
        # Assert that the number of nodes matches what was requested
        script += f'if [ "$SLURM_NNODES" -ne {self.nnodes} ]; then\n'
        script += (
            '    echo "Error: SLURM_NNODES ($SLURM_NNODES) does not match requested nodes (' + str(self.nnodes) + ')"\n'
        )
        script += "    exit 1\n"
        script += "fi\n\n"
        script += "export TREETUNEV__NNODE=$SLURM_NNODES\n"
        script += "export TREETUNEV__NUM_CPUS=$SLURM_CPUS_PER_TASK\n"
        if self.nnodes <= 1:
            script += "export TREETUNEV__LAUNCH_RAY_CLUSTER=0\n"
        else:
            script += "export TREETUNEV__LAUNCH_RAY_CLUSTER=1\n"
            script += "export TREETUNEV__RAY_HEAD_IP=$(find_head_node_ip)\n"
            script += "export TREETUNEV__RAY_HEAD_PORT=$(find_free_port)\n"
        script += "\n"

        script += f"export HF_HOME={self.experiments_dir}/hf_home\n"

        script += f"export WANDB_CACHE_DIR={self.experiments_dir}/wandb_cache_dir\n"
        if self.wandb_offline:
            script += f"export WANDB_DIR={self.experiments_dir}/{persistent_key}\n\n"
        else:
            if self.runtime_env == "singularity":
                script += "export WANDB_DIR=~\n\n"
            else:
                script += f"export WANDB_DIR={self.compute_node_storage_dir}/wandb_dir\n\n"

        # Export WANDB ENV variables
        if self.wandb_api_key is not None:
            script += f"export WANDB_API_KEY={self.wandb_api_key}\n"
        if self.wandb_project_name is not None:
            script += f"export WANDB_PROJECT={self.wandb_project_name}\n"
        if self.wandb_entity_name is not None:
            script += f"export WANDB_ENTITY={self.wandb_entity_name}\n"
        script += "unset WANDB_SERVICE\n"

        if self.wandb_offline:
            script += "export WANDB_MODE=offline\n"

        if self.hf_hub_offline:
            script += "export HF_HUB_OFFLINE=1\n"

        if self.env_vars is not None:
            for k_v in self.env_vars:
                script += f"export {k_v}\n"

        script += '\necho "Uploading contents to compute node..." \n'
        script += f"srun cp -r {tmp_exp_dir}/* {self.compute_node_storage_dir} \n\n"

        # Prefix the stdout and stderr paths with "$SLURM_JOB_ID" variable
        script += f'export base_log_path_prefix="{self.experiments_dir}/{persistent_key}/${{SLURM_JOB_ID}}"\n'

        if self.runtime_env == "singularity":
            assert self.singularity_module is not None, "singularity_module is not set"
            script += f"export SINGULARITY_CACHEDIR={self.compute_node_storage_dir}/singularity_cache\n"
            image_path = self.singularity_image_library_path / self.image_name
            script += f'echo "Copying container {image_path} to compute node..." \n'
            script += f"srun cp {image_path} {self.compute_node_storage_dir}/ \n\n"

        script += "cd $HOME\n"

        script += '\necho "Running the computation..." \n'

        if self.runtime_env == "singularity":
            script += self._build_singularity_runtime_script()
        elif self.runtime_env == "python_venv":
            script += self._build_python_venv_runtime_script()
        else:
            raise ValueError(f"Invalid runtime environment: {self.runtime_env}")

        return script

    def _build_singularity_runtime_script(self) -> str:
        """Build the Singularity runtime script based on interactive mode."""
        script = ""

        if self.interactive:
            script += f"module load {self.singularity_module}\n"
            script += f"mkdir -p $SINGULARITY_CACHEDIR\n"
            script += f"{self.singularity_cmd} shell --nv \\\n"
            script += f"\t-H {self.compute_node_storage_dir}/home:$HOME \\\n"
            script += f"\t-B {self.experiments_dir}:$HOME/experiments \\\n"
            script += f"\t-B {self.cluster_shared_storage_dir}:$HOME/{self.cluster_shared_storage_dir.name} \\\n"
            script += f"\t-B /network:/network \\\n"
            script += f"\t{self.compute_node_storage_dir}/{self.image_name} \n\n"
        else:
            script += "srun bash -c \\\n"
            script += "'"
            script += "export TREETUNEV__NODE_RANK=$SLURM_NODEID \\\n"
            script += '&& export APP_LOG_PATH_W_PREFIX="${base_log_path_prefix}.${SLURM_NODEID}" \\\n'
            script += f"&& module load {self.singularity_module} \\\n"
            script += f"&& mkdir -p $SINGULARITY_CACHEDIR \\\n"
            script += f"&& unset ROCR_VISIBLE_DEVICES \\\n"
            script += f"&& cd $HOME \\\n"
            script += f"&& {self.singularity_cmd} exec --nv \\\n"
            script += f"\t-H {self.compute_node_storage_dir}/home:$HOME \\\n"
            script += f"\t-B {self.experiments_dir}:$HOME/experiments \\\n"
            script += f"\t-B {self.cluster_shared_storage_dir}:$HOME/{self.cluster_shared_storage_dir.name} \\\n"
            script += f"\t-B /network:/network \\\n"
            script += f"\t{self.compute_node_storage_dir}/{self.image_name} \\\n"
            script += f"\t./{self.run_script_name} > ${{APP_LOG_PATH_W_PREFIX}}.log 2>&1"
            script += "'\n\n"

        return script

    def _build_python_venv_runtime_script(self) -> str:
        """Build the Python venv runtime script based on interactive mode."""
        script = ""

        if self.interactive:
            if self.httpproxy_module is not None:
                script += f"module load {self.httpproxy_module}\n"
            script += f"source {self.python_venv_library_path}/{self.image_name}\n"
            script += f"cd {self.compute_node_storage_dir}/home\n"
            script += f"ln -snf {self.experiments_dir} ./experiments\n"
            script += f"ls -lha .\n\n"
        else:
            script += "srun bash -c \\\n"
            script += "'"
            script += "export TREETUNEV__NODE_RANK=$SLURM_NODEID \\\n"
            script += '&& export APP_LOG_PATH_W_PREFIX="${base_log_path_prefix}.${SLURM_NODEID}" \\\n'
            if self.httpproxy_module is not None:
                script += f"&& module load {self.httpproxy_module} \\\n"
            script += f"&& source {self.python_venv_library_path}/{self.image_name} \\\n"
            script += f"&& cd {self.compute_node_storage_dir}/home \\\n"
            script += f"&& ln -snf {self.experiments_dir} ./experiments \\\n"
            script += f"&& ./{self.run_script_name} > ${{APP_LOG_PATH_W_PREFIX}}.log 2>&1"
            script += "'\n\n"

        return script


class ComputeCanadaCluster(SlurmComputingCluster):
    def __init__(
        self,
        **kwargs,
    ):
        account = kwargs.pop("account", "rrg-bengioy-ad")
        wandb_offline = kwargs.pop("wandb_offline", False)
        hf_hub_offline = kwargs.pop("hf_hub_offline", False)
        super().__init__(
            **kwargs,
            shared_storage_dir="~/scratch",
            account=account,
            wandb_offline=wandb_offline,
            hf_hub_offline=hf_hub_offline,
            runtime_env="python_venv",
        )

    def _create_worker_script(self) -> str:
        """Create the worker script content with CUDA environment setup."""
        worker_script = "#!/bin/bash\n\n"
        worker_script += "echo 'Copying HF_HOME to compute node...'\n"
        worker_script += f"cp -r $HF_HOME {self.compute_node_storage_dir}/home/hf_home\n"
        worker_script += f"export HF_HOME={self.compute_node_storage_dir}/home/hf_home\n"
        worker_script += "export IBV_DRIVERS=mlx5\n"
        worker_script += "unset ROCR_VISIBLE_DEVICES\n"
        worker_script += "chmod a+x run.sh\n"
        worker_script += "./run.sh\n\n"
        return worker_script


class TamiaCluster(SlurmComputingCluster):
    def __init__(
        self,
        **kwargs,
    ):
        wandb_offline = kwargs.pop("wandb_offline", True)
        hf_hub_offline = kwargs.pop("hf_hub_offline", False)
        shared_storage_dir = kwargs.pop("shared_storage_dir", "~/scratch")
        super().__init__(
            **kwargs,
            shared_storage_dir=shared_storage_dir,
            wandb_offline=wandb_offline,
            hf_hub_offline=hf_hub_offline,
            runtime_env="python_venv",
            httpproxy_module="httpproxy",
        )

    def _create_worker_script(self) -> str:
        """Create the worker script content with CUDA environment setup."""
        worker_script = "#!/bin/bash\n\n"
        worker_script += "export IBV_DRIVERS=mlx5\n"
        worker_script += "unset ROCR_VISIBLE_DEVICES\n"
        worker_script += "chmod a+x run.sh\n"
        worker_script += "./run.sh\n\n"
        return worker_script


class MilaCluster(SlurmComputingCluster):
    def __init__(self, **kwargs):
        wandb_offline = kwargs.pop("wandb_offline", False)
        hf_hub_offline = kwargs.pop("hf_hub_offline", False)
        super().__init__(
            **kwargs,
            shared_storage_dir="~/scratch",
            wandb_offline=wandb_offline,
            hf_hub_offline=hf_hub_offline,
        )

    def _create_worker_script(self) -> str:
        """Create the worker script content with CUDA environment setup."""
        worker_script = "#!/bin/bash\n\n"
        worker_script += "unset CURL_CA_BUNDLE\n"
        worker_script += "chmod a+x run.sh\n"
        worker_script += "./run.sh\n\n"
        return worker_script


class ComputeCanadaFirCluster(SlurmComputingCluster):
    def __init__(self, **kwargs):
        wandb_offline = kwargs.pop("wandb_offline", False)
        hf_hub_offline = kwargs.pop("hf_hub_offline", False)
        account = kwargs.pop("account", "rrg-bengioy-ad")
        super().__init__(
            **kwargs,
            shared_storage_dir="~/scratch",
            wandb_offline=wandb_offline,
            hf_hub_offline=hf_hub_offline,
            runtime_env="python_venv",
            account=account,
        )


class ServiceNowCluster(SlurmComputingCluster):
    def __init__(
        self,
        slurm_args: str,
        image_name: str = "latest.sif",
        wait_for_login_script: bool = False,
        scratch_dir: str = "/scratch",
        domain_name: str = "snow.research.mmteb",
        **kwargs,
    ):
        super(SlurmComputingCluster, self).__init__(**kwargs)

        self.domain_name = domain_name

        repo_dir = Path(__file__).resolve().parent.parent
        logs_dir = f"{repo_dir}/experiments/launcher_logs"
        scripts_dir = f"{repo_dir}/experiments/launcher_scripts"

        self.cluster_shared_storage_dir = Path(scratch_dir)
        self.global_project_dir = self.cluster_shared_storage_dir / self.project_name
        self.experiments_dir = self.global_project_dir / "experiments"
        self.global_code_dir = Path("/codes")
        self.compute_node_storage_dir = "$HOME"

        self.log_dir = Path(logs_dir) / f"lid_{self.launcher_id}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.script_dir = Path(scripts_dir)
        self.script_dir.mkdir(parents=True, exist_ok=True)

        self.image_name = image_name
        self.slurm_args = slurm_args

        self.wait_for_login_script = wait_for_login_script

        self.run_script_name = "entrypoint.sh"

        self.experiments_dir = self.cluster_shared_storage_dir / self.project_name / "experiments"

    def _servicenow_safe_job_name(self, name: str) -> str:
        """Sanitize job name for ServiceNow constraints.

        Constraint: must be lowercase alphanumeric with underscores allowed.
        - Lowercase all characters
        - Replace any char not in [a-z0-9_] with '_'
        - Collapse multiple '_' and trim leading/trailing '_'
        """
        name = name.lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        # Avoid empty name edge case
        return name or "job"

    def download_job_files(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)

        import wandb

        project = self.wandb_project_name
        entity = self.wandb_entity_name
        api_key = self.wandb_api_key

        overrides = {}
        if project is not None:
            overrides["project"] = project
        if entity is not None:
            overrides["entity"] = entity
        api = wandb.Api(
            overrides=overrides if len(overrides) > 0 else None,
            api_key=api_key,
        )

        if entity is not None:
            artifact_name = f"{entity}/{project}/"
        else:
            artifact_name = ""

        artifact_name += f"bundle-{self.launcher_id}:latest"
        artifact = api.artifact(artifact_name)
        artifact.download(str(output_dir))

        try:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            persistent_key = metadata["exp_name"]
        except Exception as e:
            print("Unable to load metadata.json, computing persistent_dir based on launcher_id")
            persistent_key = create_md5_hash(self.launcher_id)

        return persistent_key

    def execute_job(self, job_body):
        login_script_path, entrypoint_script_path, persistent_key = self.create_launch_script(job_body)

        if self.dry_run:
            print("Dry run, not executing job")
            print(f"Login script: {login_script_path}")
            print(f"Entrypoint script: {entrypoint_script_path}")
            job_spec_path = self.script_dir / f"{self.launcher_id}_job_spec.yaml"
            print(f"Job Spec: {job_spec_path}")
            return

        # Run the login script and capture output
        print("Preparing the job and submitting it...")
        try:
            process = subprocess.run([login_script_path], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Error running login script:", e)
            print("Raw output:", e.output)
            print("Raw stderr:", e.stderr)
            return

        # Try to parse any JSON output from the process
        try:
            output_lines = process.stdout.split("------JOB SUBMISSION OUTPUT------")[-1].strip()
            job_info = json.loads(output_lines)
        except Exception as e:
            print("Could not parse job submission output:", e)
            print("Raw output:", process.stdout)
            return

        job_id = job_info["id"]
        replica_group_id = job_info.get("replicaGroupId")

        if self.interactive:
            print("Started executing...")
            print(f"Job ID: {job_id}")
            if replica_group_id is not None:
                print(f"Replica Group ID: {replica_group_id}")
            print("\nRun the following command to attach to the job")
            print(f"$ eai job exec {job_id} /bin/bash")
            print("\nOnce you are there, run the following command to setup the environment")
            print(f"$ source {self.global_code_dir}/{self.launcher_id}/{self.run_script_name}")
        else:
            print("Started executing...")
            print(f"Job ID: {job_id}")
            if replica_group_id is not None:
                print(f"Replica Group ID: {replica_group_id}")
            print("\nRun the following command to attach to the job")
            print(f"$ eai job exec {job_id} /bin/bash")
            print("\nTo check the logs, run the following command:")
            print(f"$ eai job log -f {job_id}")

        return persistent_key

    def create_launch_script(self, job_body) -> tuple[Path, Path, str]:
        tmp_dir = get_tempfile_path()
        tmp_dir.mkdir(parents=True, exist_ok=True)

        code_dir = self.global_code_dir / self.launcher_id

        persistent_key = self.download_job_files(tmp_dir)

        # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")

        persistent_part = self._servicenow_safe_job_name(str(persistent_key))
        base_name = f"{self.launcher_id}_{persistent_part}" if persistent_part else f"{self.launcher_id}"
        job_name = self._servicenow_safe_job_name(f"{base_name}")

        entrypoint_script, used_ports = self.create_entrypoint_script(code_dir, persistent_key, job_name)
        save_and_make_executable(tmp_dir / self.run_script_name, entrypoint_script)
        save_and_make_executable(self.script_dir / f"{self.launcher_id}_entrypoint.sh", entrypoint_script)

        if self.interactive:
            cmd = shlex.split('bash -c "while true; do sleep 3600; done"')
        else:
            cmd = shlex.split(f"sh {code_dir}/{self.run_script_name}")

        jobs_spec = {
            "image": self.image_name,
            "command": cmd,
            "restartable": True,
            "options": {
                "internal-dns": {
                    "name": f"ttn-{self.launcher_id}",
                    "ports": [
                        {
                            "port": p,
                            "target-port": p,
                        }
                        for p in used_ports
                    ],
                }
            },
            "data": [
                f"{self.domain_name}.{self.project_name}_scratch:{self.cluster_shared_storage_dir}",
                f"{self.domain_name}.{self.project_name}_codes:{self.global_code_dir}",
            ],
            "name": job_name,
        }
        if self.nnodes > 1:
            jobs_spec["options"]["infiniband"] = True
            jobs_spec["resources"] = {"replicas": self.nnodes}
        if self.env_vars is not None:
            jobs_spec["environmentVars"] = self.env_vars

        login_script = self._create_login_script(tmp_dir, jobs_spec)

        login_script_path = self.script_dir / f"{self.launcher_id}_login.sh"
        save_and_make_executable(login_script_path, login_script)

        return login_script_path, self.script_dir / f"{self.launcher_id}_entrypoint.sh", persistent_key

    def _create_login_script(self, tmp_exp_dir: Path, job_spec: dict[str, Any]) -> str:
        # Save job spec as yaml file
        job_spec_path = self.script_dir / f"{self.launcher_id}_job_spec.yaml"
        with open(job_spec_path, "w") as f:
            yaml.dump(job_spec, f)

        # Check if the job has `pre_submit_script.sh`
        # pre_submit_script_path = tmp_exp_dir / "pre_submit_script.sh"
        # if pre_submit_script_path.exists():
        #     raise ValueError("pre_submit_script.sh is not supported for ServiceNow")

        script = "#!/bin/bash \n\n"
        script += "echo 'Copying job files to global code directory...'\n"
        script += f"eai data push {self.domain_name}.{self.project_name}_codes {tmp_exp_dir}:/{self.launcher_id}\n\n"
        script += f"eai data push {self.domain_name}.{self.project_name}_codes {tmp_exp_dir}:/{self.launcher_id}\n\n"

        script += "echo '------JOB SUBMISSION OUTPUT------'\n"
        script += f"eai job new --format json -f {job_spec_path} {self.slurm_args}\n"

        return script

    def create_entrypoint_script(self, code_dir: Path, persistent_key: str, job_name: str) -> tuple[str, list[int]]:
        script = "#!/bin/bash\n\n"

        script += f"\nexport HF_HOME={self.experiments_dir}/hf_home\n"
        script += f"export WANDB_DIR={self.compute_node_storage_dir}\n\n"

        # Export WANDB ENV variables
        if self.wandb_api_key is not None:
            script += f"export WANDB_API_KEY={self.wandb_api_key}\n"
        if self.wandb_project_name is not None:
            script += f"export WANDB_PROJECT={self.wandb_project_name}\n"
        if self.wandb_entity_name is not None:
            script += f"export WANDB_ENTITY={self.wandb_entity_name}\n"
        script += "unset WANDB_SERVICE\n"

        if self.hf_token is not None:
            script += f"\nexport HF_TOKEN={self.hf_token}\n\n"

        # find num requested CPU cores from self.slurm_args (using --cpu)
        args = self.slurm_args.split()
        num_cpus = None
        for i, arg in enumerate(args):
            if arg.startswith("--cpu"):
                if arg.endswith("="):
                    num_cpus = int(args[i + 1])
                else:
                    num_cpus = int(args[i + 1])
                break
        assert num_cpus is not None, "Could not find number of requested CPU cores from args"

        ray_head_port = 6379
        used_ports = [ray_head_port]

        script += f"export TREETUNEV__NNODE={self.nnodes}\n"
        script += f"export TREETUNEV__NUM_CPUS={num_cpus}\n"
        if self.nnodes == 1:
            script += f"export TREETUNEV__LAUNCH_RAY_CLUSTER=0\n"
            script += f"export TREETUNEV__NODE_RANK=0\n\n"
        else:
            script += f"export TREETUNEV__LAUNCH_RAY_CLUSTER=1\n"
            script += f"export TREETUNEV__RAY_HEAD_IP=$MASTER_ADDR\n"
            script += f"export TREETUNEV__RAY_HEAD_PORT={ray_head_port}\n"
            script += f"export TREETUNEV__NODE_RANK=$RANK\n"
            script += f"unset MASTER_ADDR\n"
            script += f"unset RANK\n\n"

        script += 'echo "Uploading contents to compute node..." \n'
        script += f"mkdir -p {self.compute_node_storage_dir}\n"
        script += f"cp -r {code_dir}/* {self.compute_node_storage_dir} \n\n"

        script += f"mkdir -p {self.experiments_dir}/{persistent_key}\n"
        script += f"ln -sfn {self.experiments_dir} {self.compute_node_storage_dir}/experiments\n"
        script += f"ln -sfn {self.cluster_shared_storage_dir} {self.compute_node_storage_dir}/{self.cluster_shared_storage_dir.name}\n\n"

        script += (
            f"export base_log_path_prefix={self.compute_node_storage_dir}/experiments/{persistent_key}/{job_name}\n"
        )
        script += 'export APP_LOG_PATH_W_PREFIX="${base_log_path_prefix}.${TREETUNEV__NODE_RANK}"\n\n'

        script += f"cd {self.compute_node_storage_dir}\n"
        script += "chmod a+x ./run.sh\n"

        pre_submit_script_path = f"{self.compute_node_storage_dir}/pre_submit_script.sh"
        script += f'if [ -f "{pre_submit_script_path}" ]; then\n'
        script += '    echo "Running pre_submit_script.sh..."\n'
        script += f'    bash "{pre_submit_script_path}"\n'
        script += "fi\n\n"

        if not self.interactive:
            script += '\necho "Running the computation..." \n'
            script += "./run.sh 2>&1 | tee ${APP_LOG_PATH_W_PREFIX}.log"

        return script, used_ports


def get_config(required_keys: list[str], available_platforms: list[str] | None = None) -> dict[str, Union[str, bool]]:
    config_path = Path(__file__).parent / ".launcher_config.json"
    if config_path.exists():
        config_ob = json.load(config_path.open())
    else:
        config_ob = {}

    if available_platforms is None:
        available_platforms = []

    key_to_message = {
        "wandb_api_key": "Enter your wandb api key",
        "conda_env_name": "Enter the name of the conda environment",
        "wandb_project_name": "Enter the name of the wandb project",
        "wandb_entity_name": "Enter the name of the wandb entity",
        "platform": "Enter the name of the platform",
    }

    for key in required_keys:
        if key in config_ob:
            continue

        from InquirerPy import inquirer

        new_config_ob = {
            k: inquirer.text(
                message=key_to_message.get(k, f"Enter {k}"),
            ).execute()
            if k != "platform"
            else inquirer.select(message=key_to_message["platform"], choices=available_platforms).execute()
            for k in [key]
        }
        config_ob.update(new_config_ob)

    with config_path.open("w") as f:
        json.dump(config_ob, f, indent=4)

    return config_ob


def launch_job(args: argparse.Namespace) -> None:
    available_platforms = ["mila", "cc", "cc_fir", "tamia", "trc"]

    if args.platform is None:
        args.platform = get_config(["platform"], available_platforms)["platform"]

    if args.platform == "mila":
        cluster_class = MilaCluster
    elif args.platform == "cc":
        cluster_class = ComputeCanadaCluster
    elif args.platform == "cc_fir":
        cluster_class = ComputeCanadaFirCluster
    elif args.platform == "tamia":
        cluster_class = TamiaCluster
    elif args.platform in ["servicenow", "trc"]:
        cluster_class = ServiceNowCluster
    else:
        raise ValueError()

    cluster_kwargs = vars(args)
    for k in list(cluster_kwargs.keys()):
        if cluster_kwargs[k] is None:
            del cluster_kwargs[k]

    required_keys = ["wandb_api_key", "wandb_project_name", "wandb_entity_name", "hf_token"]

    config = get_config(required_keys)

    clstr_args = copy.deepcopy(cluster_kwargs)
    clstr_args.update({"launcher_id": args.bundle, "config": config})
    if "project_name" not in clstr_args and "wandb_project_name" in config:
        clstr_args["project_name"] = config["wandb_project_name"]

    clstr = cluster_class(**clstr_args)

    clstr.execute_job(None)


def get_queued_jobs() -> list[tuple[str, str, str]]:
    user = os.environ.get("USER")
    cmd = f"squeue -u {user} -o %A,%j,%T --noheader"
    output = subprocess.check_output(shlex.split(cmd)).decode("utf-8")
    jobs = []
    for line in output.splitlines():
        job_id, job_name, state = line.split(",")
        launcher_id = job_name.split("_compute.sh")[0]
        jobs.append((job_id, launcher_id, state))
    return jobs


def main(args):
    if args.bundle is not None:
        if "," not in args.bundle:
            bundles = [args.bundle]
        else:
            bundles = args.bundle.split(",")
            bundles = [b.strip() for b in bundles if b != ""]
    else:
        bundles = [None]

    queued_jobs = []
    if args.nodup:
        try:
            queued_jobs = get_queued_jobs()
            print(f"Queued jobs:")
            from pprint import pprint

            pprint(queued_jobs)
        except subprocess.CalledProcessError as e:
            print("Could not get queued jobs")
            print(e)

    already_launched = set([job[1] for job in queued_jobs])

    print("Bundles:", bundles)
    for bundle in bundles:
        if args.nodup and bundle in already_launched:
            print(f"Skipping {bundle} because it is already queued")
            continue

        args.bundle = bundle

        launch_job(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment runner")

    parser.add_argument("bundle", metavar="EXP_KEY", nargs="?", type=str, help="Wandb ID")

    parser.add_argument(
        "-p",
        "--platform",
        metavar="PLATFORM",
        type=str,
        choices=["mila", "tamia", "cc", "cc_fir", "servicenow"],
        help="The computation platform we're running the experiment",
    )

    parser.add_argument(
        "-s",
        "--slurm_args",
        metavar="ARGS",
        type=str,
        default="--gres=gpu:1",
        help="Slurm args",
    )

    parser.add_argument("-i", "--image_name", metavar="IMAGE", type=str, help="Container Image")

    parser.add_argument(
        "--images_dir",
        metavar="DIR",
        type=str,
        help="Container Images Directory (only needed for singularity)",
    )

    parser.add_argument(
        "--shared_storage_dir",
        metavar="DIR",
        type=str,
        help="Path to cluster's shared storage between compute nodes and login node",
    )

    parser.add_argument(
        "--compute_storage_dir",
        metavar="DIR",
        type=str,
        help="Path to on-device storage on compute nodes",
    )

    parser.add_argument(
        "--account",
        metavar="ACCOUNT",
        type=str,
        help="Slurm account (only needed for CC)",
    )

    parser.add_argument(
        "--scripts-dir",
        metavar="DIR",
        type=str,
        help="Directory to output generated job scripts",
    )

    parser.add_argument(
        "--logs-dir",
        metavar="DIR",
        type=str,
        help="Directory to store jobs' log",
    )

    parser.add_argument(
        "--env_vars",
        metavar="ENVS",
        type=str,
        help="Environment variables passed to the container, e.g. X1=V1,x2=V2",
    )

    parser.add_argument(
        "--nodup",
        action="store_true",
        help="Do not run already queued experiments",
        default=False,
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Print queued experiments' info",
        default=False,
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run the job interactively",
        default=False,
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't execute the jobs (Useful for testing and debugging).",
        default=False,
    )

    parser.add_argument(
        "-n",
        "--num_submissions",
        metavar="NUM",
        type=int,
        default=1,
        help="Number of job to submit",
    )

    parser.add_argument(
        "--wandb_api_key",
        metavar="API_KEY",
        type=str,
        help="Wandb API key",
    )

    parser.add_argument(
        "--wandb_project_name",
        metavar="PROJECT_NAME",
        type=str,
        help="Wandb project name",
    )

    parser.add_argument(
        "--wandb_entity_name",
        metavar="ENTITY_NAME",
        type=str,
        help="Wandb entity name",
    )

    parser.add_argument(
        "--hf_token",
        metavar="TOKEN",
        type=str,
        help="Hugging Face token",
    )

    parser.add_argument(
        "--dist_nodes",
        metavar="NODES",
        type=int,
        help="Number of nodes for distributed training",
    )

    args = parser.parse_args()

    main(args)
