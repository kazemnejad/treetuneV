import argparse
import hashlib
import json
import os
import random
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from wandb import env as wandb_env

os.environ[wandb_env.SILENT] = "true"
os.environ[wandb_env.DISABLE_CODE] = "true"


def create_md5_hash(inp: str):
    # Create MD5 hash object
    md5 = hashlib.md5()
    # Update the hash with the string
    md5.update(inp.encode("utf-8"))
    # Get the hexadecimal representation of the hash
    return md5.hexdigest()


def get_repo_dir() -> Path:
    return Path(__file__).parent.parent


def collect_repo_directory(
    directory: Path,
    exclude_fn: Optional[Callable[[Path], bool]] = None,
    include_fn: Optional[Callable[[Path], bool]] = None,
    relative_to: Optional[Path] = None,
) -> list[tuple[Path, Path]]:
    if include_fn is None:
        include_fn = lambda path: True

    if exclude_fn is None:
        exclude_fn = lambda path: False

    if relative_to is None:
        relative_to = directory.parent.absolute()
    else:
        assert relative_to.is_absolute()

    # Use glob to fetch all files recursively
    paths = directory.absolute().glob("**/*")

    files = [
        (path, path.relative_to(relative_to))
        for path in paths
        if not path.is_dir() and include_fn(path) and not exclude_fn(path)
    ]

    return files


def save_as_temp(content: str, make_executable: bool = False) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        if make_executable:
            os.fchmod(f.fileno(), 0o755)
        return Path(f.name)


class Experiment:
    @classmethod
    def get_arg_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--config",
            metavar="CONFIG",
            type=str,
            help="Config file name",
        )

        parser.add_argument(
            "--commands",
            metavar="cmd -a -b[,cmd -c -d]",
            type=str,
            help="Experiment commands",
            required=True,
        )

        parser.add_argument(
            "--env-vars",
            metavar="KEY=VAL[,KEY=VAL]",
            type=str,
            help="Experiment environment variables",
        )

        parser.add_argument(
            "--tags",
            metavar="VAL[,VAL]",
            type=str,
            help="Experiment tags",
        )

        parser.add_argument(
            "--num_seeds",
            type=int,
            help="Number of seeds to run",
            default=1,
        )

        parser.add_argument(
            "-i",
            "--idx",
            metavar="IDX",
            type=str,
            help="Experiment Idx",
        )

        parser.add_argument(
            "--group",
            metavar="GROUP",
            type=str,
            help="Wandb run group name",
        )

        parser.add_argument(
            "--group_postfix",
            metavar="GROUP_POSTFIX",
            type=str,
            help="Wandb run group postfix",
        )

        parser.add_argument(
            "--extra-pip-packages",
            metavar="PKG[,PKG]",
            type=str,
            help="Extra pip packages",
            default=None,
        )

        parser.add_argument(
            "--pre-submit",
            metavar="SCRIPT",
            type=str,
            help="Pre-submit script to run on the login node",
            default=None,
        )

        parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Pass args after `--` to the target")

        return parser

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.num_seeds = args.num_seeds

        # Config files
        assert args.config is not None, "No config files specified"
        self.config = args.config.strip()
        # assert os.path.exists(self.config), f"Config file {self.config} does not exist"

        # Parse env vars
        self.env_vars = {}
        if args.env_vars is not None:
            for env_var in args.env_vars.split(","):
                key, val = env_var.split("=")
                self.env_vars[key] = val

    def upload_to_cloud(self) -> tuple[str, str]:
        import wandb

        settings = wandb.Settings()
        if hasattr(settings, "update"):
            settings.update(
                disable_code=True,
                disable_git=True,
                silent=True,
                _save_requirements=False,
                _disable_meta=True,
            )

        temp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Log the command used to launch this script
        interpreter = sys.executable
        escaped_args = [shlex.quote(arg) for arg in sys.argv]
        upload_command = " ".join([interpreter] + escaped_args)

        run = wandb.init(
            project=self.get_project(),
            entity=self.get_entity(),
            dir=temp_dir,
            group=self.get_group(),
            name=self.get_uploader_run_name(),
            config={"upload_command": upload_command},
            mode="online",
            force=True,
            save_code=False,
            settings=settings,
            job_type="uploader",
            id=self.get_uploader_exp_idx(),
            resume="allow",
            tags=self.get_tags(),
        )
        run_id = run.id

        # Create Experiment Artifact
        artifact = wandb.Artifact(
            name=f"bundle-{run_id}",
            type="code",
        )

        # Add files
        experiment_files = self.get_experiment_files(run_id, self.get_experiment_unique_id())
        for file_path, artifact_path in experiment_files:
            artifact.add_file(str(file_path), name=artifact_path)

        run.log_artifact(artifact)
        run.finish()

        return run_id, run.url

    def get_experiment_files(self, launcher_run_id, experiment_unique_id) -> list[tuple[Path, str]]:
        files = []

        # Add project source files (src, configs, scripts, notebooks)
        files += self.get_repo_source_files()

        # Generate job.sh
        job_sh_content = self.generate_run_sh(launcher_run_id, experiment_unique_id)
        files += [(save_as_temp(job_sh_content, make_executable=True), "run.sh")]

        # Generate metadata.json
        metadata_content = self.generate_metadata_content()
        files += [(save_as_temp(metadata_content), "metadata.json")]

        # Add pre-submit script
        if self.args.pre_submit is not None:
            pre_submit_script = Path(self.args.pre_submit).read_text()
            files += [(save_as_temp(pre_submit_script), "pre_submit_script.sh")]

        return files

    def generate_run_sh(self, launcher_run_id: str, experiment_unique_id: str) -> str:
        job_sh = self._generate_job_header()
        job_sh += self._generate_job_env_vars()
        job_sh += self._generate_job_body(experiment_unique_id, launcher_run_id)
        job_sh += self._generate_job_footer()

        return job_sh

    def _generate_job_header(self) -> str:
        # Bash script header
        job_sh = "#!/bin/bash\n"
        job_sh += "set -e\n\n"  # Make sure to exit on error

        if self.args.extra_pip_packages is not None:
            pip_packages = self.args.extra_pip_packages.split(",")
            for pip_package in pip_packages:
                job_sh += f"pip install --user {pip_package}\n"

        # Add local Python site packages to PYTHONPATH
        python_versions = ["3.10", "3.11", "3.12"]
        for python_version in python_versions:
            job_sh += f"export PYTHONPATH=$HOME/.local/lib/python{python_version}/site-packages/:$PYTHONPATH\n"

        job_sh += "NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)\n"
        job_sh += "\n"

        return job_sh

    def _generate_job_body(self, experiment_unique_id, launcher_run_id):
        job_sh = f"export WANDB_RUN_GROUP={self.get_group()}\n"
        job_sh += f"export WANDB_TAGS=launched_by_{launcher_run_id}"
        if len(self._get_tags_as_str()) > 0:
            job_sh += f",{self._get_tags_as_str()}"
        job_sh += "\nexport WANDB_JOB_TYPE=job\n\n"

        seeds = self._get_random_seeds()
        wandb_run_id_prefix = create_md5_hash(experiment_unique_id)

        job_sh += "export VERL_LOGGING_LEVEL=INFO\n"
        job_sh += f"export TREETUNEV__EXP_ROOT_DIR=experiments/{experiment_unique_id}\n"
        job_sh += "export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS\n"
        job_sh += "ln -snf $TREETUNEV__EXP_ROOT_DIR _app_dir\n\n"

        job_sh += "\n####################################\n"
        job_sh += "# Launch Ray Cluster if needed \n"
        job_sh += "####################################\n"
        job_sh += "launch_ray=${TREETUNEV__LAUNCH_RAY_CLUSTER:-0}\n"
        job_sh += (
            "if [ $launch_ray -eq 1 ]; then\n"
            '  if [ "${TREETUNEV__NODE_RANK:-0}" -eq 0 ]; then\n'
            '    echo "Starting Ray head node on $TREETUNEV__RAY_HEAD_IP:$TREETUNEV__RAY_HEAD_PORT"\n'
            '    ray start --head --port="$TREETUNEV__RAY_HEAD_PORT" \\\n'
            '      --num-cpus "$TREETUNEV__NUM_CPUS" --num-gpus "$NUM_GPUS"\n'
            '    sleep 10\n'
            '  else\n'
            '    sleep 10\n'
            '    echo "[worker] waiting for head at $TREETUNEV__RAY_HEAD_IP:$TREETUNEV__RAY_HEAD_PORT"\n'
            '    ray start --address="${TREETUNEV__RAY_HEAD_IP}:${TREETUNEV__RAY_HEAD_PORT:-6379}" \\\n'
            '      --num-cpus "$TREETUNEV__NUM_CPUS" --num-gpus "$NUM_GPUS" --block\n'
            '    exit 0\n'
            '  fi\n'
            "fi\n\n"
        )

        job_sh += "\n####################################\n"
        job_sh += "# Run the experiment \n"
        job_sh += "####################################\n"
        job_sh += (
            f"for SEED in {' '.join([str(s) for s in seeds])}; do\n"
            "\texport TREETUNEV__SEED=$SEED\n"
            '\texport TREETUNEV__EXP_NAME="${WANDB_RUN_GROUP}@sd_${SEED}"\n'
            f"\texport WANDB_RUN_ID={wandb_run_id_prefix}__seed_$SEED\n\n"
        )
        for cmd in self.args.commands.split(","):
            job_sh += self._command_to_bash_str(cmd, self.config, prefix="\t")
        job_sh += "done\n"
        return job_sh

    def _generate_job_footer(self):
        job_sh = "\necho 'Job finished'\n"
        return job_sh

    def _generate_job_env_vars(self) -> str:
        job_sh = "\n"
        for key, val in self.env_vars.items():
            job_sh += f"export {key}={val}\n"

        return job_sh

    def get_group(self) -> Optional[str]:
        if self.args.group is not None:
            return self.args.group

        group = f"{self._generate_config_postfix()}"
        if self.args.group_postfix is not None:
            group += f"-{self.args.group_postfix}"

        return group

    def get_experiment_unique_id(self) -> str:
        return self.get_group()

    def get_uploader_run_name(self) -> Optional[str]:
        return "Uploader"

    def get_uploader_exp_idx(self) -> Optional[str]:
        return self.args.idx

    def get_tags(self) -> Optional[list[str]]:
        default_tags = [f"ExpType__{self.__class__.__name__}"]
        if self.args.tags is not None:
            return [tag.strip() for tag in self.args.tags.split(",")] + default_tags
        return default_tags

    def generate_metadata_content(self) -> str:
        metadata = {"exp_name": self.get_experiment_unique_id()}
        return json.dumps(metadata, indent=4)

    def get_repo_source_files(self) -> list[tuple[Path, str]]:
        exclude_pycache_fn = lambda path: path.name.endswith(".pyc") or path.name.endswith("__pycache__")

        files = [
            (get_repo_dir() / "pyproject.toml", "pyproject.toml"),
            (get_repo_dir() / "setup.py", "setup.py"),
            (get_repo_dir() / "requirements.txt", "requirements.txt"),
            (get_repo_dir() / "requirements_sglang.txt", "requirements_sglang.txt"),
            *collect_repo_directory(get_repo_dir() / "verl", exclude_fn=exclude_pycache_fn),
            *collect_repo_directory(get_repo_dir() / "tests", exclude_fn=exclude_pycache_fn),
            *collect_repo_directory(get_repo_dir() / "scripts", exclude_fn=exclude_pycache_fn),
            *collect_repo_directory(get_repo_dir() / "examples", exclude_fn=exclude_pycache_fn),
            *collect_repo_directory(get_repo_dir() / "recipe", exclude_fn=exclude_pycache_fn),
        ]

        return files

    def get_entity(self) -> Optional[str]:
        if "WANDB_ENTITY" in os.environ:
            # Wandb client will handle this
            return None

        wandb_account_file = get_repo_dir() / ".wandb_account.json"
        if wandb_account_file.exists():
            with wandb_account_file.open("r") as f:
                wandb_account = json.load(f)
            entity = wandb_account["entity"]
        else:
            entity = None

        return entity

    def get_project(self) -> Optional[str]:
        if "WANDB_PROJECT" in os.environ:
            # Wandb client will handle this
            return None

        wandb_account_file = get_repo_dir() / ".wandb_account.json"
        if wandb_account_file.exists():
            with wandb_account_file.open("r") as f:
                wandb_account = json.load(f)
            project = wandb_account["project"]
        else:
            project = None

        return project

    def _generate_config_postfix(self) -> str:
        return self.config

    def _get_tags_as_str(self) -> str:
        if self.get_tags() is not None:
            return ",".join(self.get_tags())
        return ""

    def _get_configs_as_str(self):
        return '"' + ",\\\n".join(self.config) + '"'

    def _command_to_bash_str(
        self,
        cmd: str,
        configs_str: str,
        prefix: str = "",
    ) -> str:
        extra_argv = self.args.extra_args
        if extra_argv and extra_argv[0] == "--":
            extra_argv = extra_argv[1:]
        else:
            extra_argv = []

        bash = f'{prefix}python -m {cmd} --config-name "{configs_str}"'
        if len(extra_argv) > 0:
            argv_str = f" \\\n{prefix}\t".join([shlex.quote(arg) for arg in extra_argv])
            bash += f" \\\n{prefix}\t{argv_str}\n"

        bash += "\n"

        return bash

    def _get_random_seeds(self) -> list[int]:
        """
        Randomly Sample seeds
        """
        rnd = random.Random(42)
        return [rnd.randint(1000, 2**32 - 1) for _ in range(self.num_seeds)]


EXPERIMENT_TYPE_TO_CLASS = {
    "default": Experiment,
}


def main():
    parser = argparse.ArgumentParser(description="Upload experiment to cloud")
    parser.add_argument(
        "--type",
        type=str,
        help="Type of experiment to run",
        choices=EXPERIMENT_TYPE_TO_CLASS.keys(),
        default="default",
    )

    # Accept all other arguments
    args, _ = parser.parse_known_args()
    experiment_class = EXPERIMENT_TYPE_TO_CLASS[args.type]
    new_parser = argparse.ArgumentParser(parents=[parser, experiment_class.get_arg_parser()], add_help=False)
    args = new_parser.parse_args()

    experiment = experiment_class(args)
    experiment_id, experiment_url = experiment.upload_to_cloud()

    print("========================================")
    print(f"Uploaded Experiment ID: {experiment_id}\n")
    print(f"Uploaded Experiment URL: {experiment_url}\n")


if __name__ == "__main__":
    # Read wandb_api_key
    if "WANDB_API_KEY" not in os.environ:
        wandb_api_key_file = get_repo_dir() / ".wandb_api_key.json"
        if wandb_api_key_file.exists():
            with wandb_api_key_file.open("r") as f:
                wandb_account = json.load(f)
            os.environ["WANDB_API_KEY"] = wandb_account["key"]
            print("WANDB_API_KEY set from .wandb_api_key.json")
    main()
