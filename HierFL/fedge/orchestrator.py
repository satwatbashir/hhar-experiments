# orchestrator.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silence TensorFlow warnings

import sys
import subprocess
import time
import toml
import json
import shutil
from pathlib import Path
import logging
import datetime
import numpy as np
import atexit
import signal
logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")

# -----------------------------------------------------------------------------
#  Create a unique run directory and configure Flower JSON logging so that
#  HierFL produces the same communication CSV structure as the baseline
# -----------------------------------------------------------------------------
RUN_ID = os.getenv("FL_RUN_ID") or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = (Path(__file__).resolve().parent / "runs" / RUN_ID).resolve()
if RUN_DIR.exists():
    shutil.rmtree(RUN_DIR)
(RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
# Expose to every subprocess (cloud, leaf servers, clients, etc.)
os.environ.setdefault("RUN_DIR", RUN_DIR.as_posix())
# Ensure Flower logs metrics in JSON (required for bytes_up/down extraction)
os.environ.setdefault("FLWR_LOGGING_FORMAT", "json")
os.environ.setdefault("FLWR_LOGGING", "json")

# Get FL_SEED for reproducibility and metrics folder organization
FL_SEED = os.getenv("FL_SEED", "42")
os.environ.setdefault("FL_SEED", FL_SEED)

from fedge.utils import fs
from fedge.partitioning import write_partitions  # noqa: E402
from fedge.task import load_hhar_windows_with_users  # noqa: E402
from fedge.task import load_data  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
#  Read **all** of our hierarchy config from pyproject.toml
# ──────────────────────────────────────────────────────────────────────────────

project_root = Path(__file__).resolve().parent
package_dir = project_root / "fedge"

cfg = toml.load(project_root / "pyproject.toml")
hier = cfg["tool"]["flwr"]["hierarchy"]
raw_cps = hier["clients_per_server"]

# Optional env overrides for quick smoke tests
_num_servers_override = os.environ.get("NUM_SERVERS_OVERRIDE")
_cps_override = os.environ.get("CLIENTS_PER_SERVER_OVERRIDE")

if _num_servers_override and _cps_override:
    # Use a uniform clients-per-server for the overridden number of servers
    NUM_SERVERS = int(_num_servers_override)
    CLIENTS_PER_SERVER_LIST = [int(_cps_override)] * NUM_SERVERS
else:
    if isinstance(raw_cps, list):
        CLIENTS_PER_SERVER_LIST = raw_cps
        NUM_SERVERS = len(CLIENTS_PER_SERVER_LIST)
    else:
        NUM_SERVERS = hier.get("num_servers", int(raw_cps))
        CLIENTS_PER_SERVER_LIST = [int(raw_cps)] * NUM_SERVERS
raw_global_rounds = hier["global_rounds"]
# Allow overriding via env for smoke tests (keeps default behavior otherwise)
GLOBAL_ROUNDS = int(os.environ.get("GLOBAL_ROUNDS_OVERRIDE", raw_global_rounds))
SERVER_ROUNDS_PER_GLOBAL = hier["server_rounds_per_global"]
CLOUD_PORT = hier["cloud_port"]
# No Dirichlet parameters needed for HHAR user-based partitioning

app_cfg = cfg["tool"]["flwr"]["app"]["config"]
TOTAL_LEAF_ROUNDS = app_cfg["num-server-rounds"]
FRACTION_FIT = app_cfg["fraction-fit"]
FRACTION_EVAL = app_cfg.get("fraction-evaluate", 1.0)
raw_local_steps = app_cfg["local-epochs"]
# Allow overriding via env for smoke tests (keeps default behavior otherwise)
LOCAL_STEPS = int(os.environ.get("LOCAL_STEPS_OVERRIDE", raw_local_steps))

# ──────────────────────────────────────────────────────────────────────────────
#  Timeouts, maximum wait attempts, etc.
# ──────────────────────────────────────────────────────────────────────────────

SERVER_TIMEOUT = SERVER_ROUNDS_PER_GLOBAL * 60 * 10 + 60  # ~10 minutes per leaf‐round + buffer
PROXY_TIMEOUT = 60 * 10                                        # 10 minutes
CLOUD_TIMEOUT = (SERVER_ROUNDS_PER_GLOBAL * GLOBAL_ROUNDS) * 60 * 10 + 300  # ~10min/round + buffer

MAX_SERVER_WAIT_ATTEMPTS = 4000000  # 1 second per attempt
MAX_PROXY_WAIT_ATTEMPTS = float('inf')   # wait indefinitely
MAX_CLOUD_WAIT_ATTEMPTS =  3000000   # 1 second per attempt


# ──────────────────────────────────────────────────────────────────────────────
#  Generate hierarchical partition file (once per fresh run)
# ──────────────────────────────────────────────────────────────────────────────

rounds_dir = project_root / "rounds"
if rounds_dir.exists() and any(rounds_dir.iterdir()):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = project_root / "archive" / ts
    archive_dir.parent.mkdir(exist_ok=True)
    shutil.move(str(rounds_dir), archive_dir)
    logger.info(f"Archived previous rounds directory to {archive_dir}")
rounds_dir.mkdir(exist_ok=True, parents=True)

parts_json = rounds_dir / "partitions.json"
if not parts_json.exists():
    logger.info("Creating user-based partitions for HHAR (no Dirichlet)…")

    # Load HHAR windows with per-window user IDs
    X_all, y_all, users_all = load_hhar_windows_with_users(
        data_root="hhar/Activity recognition exp",
        use_watches=True,
        sample_rate_hz=50,
        window_seconds=2,
        window_stride_seconds=1,
    )
    # Determine unique users and sort for determinism
    unique_users = sorted(list({u for u in users_all}))
    # Validate we have enough users to fill all clients
    total_clients = sum(CLIENTS_PER_SERVER_LIST)
    assert len(unique_users) >= total_clients, (
        f"Need >= {total_clients} users; found {len(unique_users)}"
    )

    # Assign one user per client in order across servers
    mapping = {}
    user_idx = 0
    # Build index lists per user for quick lookup
    from collections import defaultdict
    user_to_indices = defaultdict(list)
    for idx, u in enumerate(users_all):
        user_to_indices[u].append(int(idx))

    for sid, cps in enumerate(CLIENTS_PER_SERVER_LIST):
        client_map = {}
        for cid in range(cps):
            u = unique_users[user_idx]
            client_map[str(cid)] = user_to_indices[u]
            user_idx += 1
        mapping[str(sid)] = client_map

    write_partitions(parts_json, mapping)
    logger.info(f"Wrote {parts_json} with user-based partitions (one user per client)")

# Expose to subprocesses
os.environ["PARTITIONS_JSON"] = str(parts_json)

# Create necessary directories
global_rounds_dir = fs.get_global_rounds_dir(project_root)
global_rounds_dir.mkdir(exist_ok=True, parents=True)

leaf_rounds_dir = fs.get_leaf_rounds_dir(project_root)
leaf_rounds_dir.mkdir(exist_ok=True, parents=True)

signals_dir = fs.get_signals_dir(project_root)
signals_dir.mkdir(exist_ok=True, parents=True)

# Delete old signal files at the root level
for p in project_root.glob("*_complete.signal"):
    p.unlink()
for p in project_root.glob("*_started.signal"):
    p.unlink()

# Clean up any existing signal files in the signals directory
for p in signals_dir.glob("*.signal"):
    p.unlink()

# ──────────────────────────────────────────────────────────────────────────────
#  Track subprocesses so we can kill them if necessary
# ──────────────────────────────────────────────────────────────────────────────

active_processes = []         # list of (name, Popen)
leaf_server_processes = []    # list of (sid, Popen)
leaf_client_processes = []    # list of (sid, cid, Popen)
proxy_client_processes = []   # list of (sid, Popen)
cloud_proc = None  # reference to the running cloud server process

# ──────────────────────────────────────────────────────────────────────────────
#  Cleanup handler: terminate any child processes that are still running when
#  the orchestrator exits (normal exit, Ctrl-C, or SIGTERM)
# ──────────────────────────────────────────────────────────────────────────────

def _cleanup():
    """Force-kill any subprocesses started by the orchestrator."""
    global cloud_proc
    # Leaf servers
    for _sid, proc in leaf_server_processes:
        if proc.poll() is None:
            proc.kill()
    # Leaf clients
    for _sid, _cid, proc in leaf_client_processes:
        if proc.poll() is None:
            proc.kill()
    # Proxy clients
    for _sid, proc in proxy_client_processes:
        if proc.poll() is None:
            proc.kill()
    # Cloud server
    if cloud_proc and cloud_proc.poll() is None:
        cloud_proc.kill()

# Register cleanup on interpreter exit and on termination signals
atexit.register(_cleanup)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers for signal files
# ──────────────────────────────────────────────────────────────────────────────

def get_global_round_signal_path(round_num: int) -> Path:
    """
    Orchestrator expects the cloud to write:
      rounds/global/round_{round_num}/complete.signal
    when global round #round_num completes.
    """
    return fs.get_global_round_dir(project_root, round_num) / "complete.signal"

def get_global_model_path(round_num: int) -> Path:
    """
    Get the path to the global model for a specific round.
    Uses consistent zero-based indexing for round numbers.
    """
    return fs.get_global_round_dir(project_root, round_num) / "model.pkl"

# ──────────────────────────────────────────────────────────────────────────────
#  1) Start the cloud aggregator (cloud_flower.py)
# ──────────────────────────────────────────────────────────────────────────────

def start_cloud_server(global_round: int):
    """
    Launch `python fedge/cloud_flower.py` once, with the environment set to point
    at GLOBAL_ROUND = 0 (meaning "about to begin global‐round 1").
    Then wait for its "cloud_started.signal" to show up under signals/.
    """
    global cloud_proc, active_processes

    logger.info(f"Starting cloud aggregator for GLOBAL ROUND {global_round} on port {CLOUD_PORT}")

    if cloud_proc and cloud_proc.poll() is None:
        logger.info("Cloud server already running")
        return

    # First, create the signals directory if it doesn't exist
    signals_dir = fs.get_signals_dir(project_root)
    signals_dir.mkdir(parents=True, exist_ok=True)

    cloud_env = os.environ.copy()
    # So that cloud_flower can "import fedge.*"
    cloud_env["PYTHONPATH"] = str(project_root) + os.pathsep + cloud_env.get("PYTHONPATH", "")
    cloud_env["SERVER_ID"] = "cloud_server"
    # Tell the cloud "we're about to do global‐round 1, so treat this as round 0 start."
    cloud_env["GLOBAL_ROUND"] = str(global_round)
    # Tell cloud_flower.py to use the new directory structure
    cloud_env["USE_NEW_DIR_STRUCTURE"] = "1"

    cloud_cmd = [
        sys.executable,
        str(package_dir / "cloud_flower.py"),
    ]

    cloud_proc = subprocess.Popen(cloud_cmd, cwd=project_root, env=cloud_env)
    active_processes.append(("cloud_server", cloud_proc))

    # Wait up to 60 s for cloud_started.signal in signals/
    logger.info("Waiting for cloud server to start…")
    cloud_start_file = signals_dir / "cloud_started.signal"
    start_time = time.time()
    while time.time() - start_time < 60:
        if cloud_start_file.exists():
            logger.info("Cloud server started successfully")
            return
        if cloud_proc.poll() is not None:
            logger.error(f"Cloud server exited with code {cloud_proc.returncode}")
            break
        time.sleep(1)

    logger.warning("Did not see cloud_started.signal — continuing anyway…")

# ──────────────────────────────────────────────────────────────────────────────
#  2) Launch leaf servers for one global round
# ──────────────────────────────────────────────────────────────────────────────

def launch_leaf_servers(global_round: int, start_round: int) -> int:
    """
    For a given global_round (0-based in code, 1-based for directories),
    (a) compute how many local rounds each leaf must run
        = SERVER_ROUNDS_PER_GLOBAL, except possibly for the very last global round.
    (b) spawn NUM_SERVERS copies of leaf_server.py with the right flags
    (c) spawn CLIENTS_PER_SERVER leaf_client.py processes under each leaf server.
    Return "rounds_to_run"—how many leaf‐rounds each server will do in this global pass.
    """

    # If this is the last global round, we might have fewer leaf‐rounds left
    is_last_global = (global_round == GLOBAL_ROUNDS - 1)
    
    rounds_to_run = SERVER_ROUNDS_PER_GLOBAL

    # If not the first global round, pick up the previous global model
    # Note: We use global_round+1 for directories (1-indexed)
    initial_model_path = None
    if global_round > 0:
        # Use the zero-indexed global round number for file path
        cand = get_global_model_path(global_round)
        if cand.exists():
            initial_model_path = cand

    logger.info(
        f"Launching {NUM_SERVERS} leaf servers "
        f"for GLOBAL ROUND {global_round}/{GLOBAL_ROUNDS-1}:"
        f" each will run {rounds_to_run} local rounds."
    )

    # Clear any old process references
    leaf_server_processes.clear()
    leaf_client_processes.clear()

    for sid in range(NUM_SERVERS):
        n_cli = CLIENTS_PER_SERVER_LIST[sid]
        port = 5000 + sid
        logger.info(f"Starting Leaf Server {sid} on port {port}")

        srv_env = os.environ.copy()
        srv_env["PYTHONPATH"] = str(project_root) + os.pathsep + srv_env.get("PYTHONPATH", "")
        srv_env["SERVER_ID"] = str(sid)
        # We send the 0-based global_round to the server via environment
        srv_env["GLOBAL_ROUND"] = str(global_round)
        # Tell the server to use the new directory structure
        srv_env["USE_NEW_DIR_STRUCTURE"] = "1"  

        srv_cmd = [
            sys.executable,
            str(package_dir / "leaf_server.py"),
            "--server_id",      str(sid),
            "--clients_per_server", str(n_cli),
            "--num_rounds",     str(rounds_to_run),
            "--fraction_fit",   str(FRACTION_FIT),
            "--fraction_evaluate", str(FRACTION_EVAL),
            "--port",           str(port),
            "--global_round",   str(global_round),  # 0-indexed in code
            "--start_round",    str(start_round + 1),  # just for logging inside leaf_server
            # Add the 1-indexed global round for directory structure
            "--dir_round",      str(global_round + 1)
        ]
        if initial_model_path is not None:
            srv_cmd += ["--initial_model_path", str(initial_model_path)]

        server_proc = subprocess.Popen(srv_cmd, cwd=project_root, env=srv_env)
        active_processes.append((f"leaf_server_{sid}_gr{global_round}", server_proc))
        leaf_server_processes.append((sid, server_proc))

        # Give the leaf server one second to bind to its port
        time.sleep(5)

        logger.info(f"Starting {n_cli} leaf‐clients for Leaf Server {sid}")
        for cid in range(n_cli):
            cli_env = os.environ.copy()
            cli_env["PYTHONPATH"] = str(project_root) + os.pathsep + cli_env.get("PYTHONPATH", "")
            cli_env["SERVER_ID"] = str(sid)
            cli_env["CLIENT_ID"] = f"leaf_{sid}_client_{cid}_gr{global_round}"

            # Each client gets its own HF cache folder
            hf_cache_subdir = project_root / f"hf_cache/leaf_{sid}_{cid}"
            hf_cache_subdir.mkdir(parents=True, exist_ok=True)
            cli_env["HF_DATASETS_CACHE"] = str(hf_cache_subdir)

            cli_cmd = [
                sys.executable,
                str(package_dir / "leaf_client.py"),
                "--partition_id",   str(cid),
                "--num_partitions", str(n_cli),
                "--dataset_flag",   "hhar",          # always "hhar" for HHAR dataset
                "--local_epochs",   str(LOCAL_STEPS),
                "--server_addr",    f"127.0.0.1:{port}",
            ]

            client_proc = subprocess.Popen(cli_cmd, cwd=project_root, env=cli_env)
            active_processes.append((f"leaf_client_{sid}_{cid}_gr{global_round}", client_proc))
            leaf_client_processes.append((sid, cid, client_proc))
            time.sleep(0.2)

    return rounds_to_run

# ──────────────────────────────────────────────────────────────────────────────
#  3) Start Proxy Clients for this global round
# ──────────────────────────────────────────────────────────────────────────────

def start_proxy_clients(global_round: int):
    """
    Launch proxy clients (one per server), which upload local server models to cloud.
    Uses the flat directory structure where each server has its own directory.
    """
    global proxy_client_processes, active_processes

    logger.info(f"Launching {NUM_SERVERS} proxy clients to upload models to cloud")

    # Clear any existing proxy process references
    proxy_client_processes.clear()

    for sid in range(NUM_SERVERS):
        logger.info(f"Starting proxy client {sid}")

        proxy_env = os.environ.copy()
        # Ensure fedge is importable
        proxy_env["PYTHONPATH"] = str(project_root) + os.pathsep + proxy_env.get("PYTHONPATH", "")
        proxy_env["SERVER_ID"] = str(sid)
        proxy_env["PROXY_ID"] = str(sid)
        proxy_env["GLOBAL_ROUND"] = str(global_round)  # 0-indexed in code
        # Let proxies know how many server rounds this cloud instance will run
        proxy_env["TOTAL_SERVER_ROUNDS_THIS_CLOUD"] = str(SERVER_ROUNDS_PER_GLOBAL)
        
        proxy_cmd = [
            sys.executable,
            str(package_dir / "proxy_client.py"),
            "--server_id",      str(sid),
            "--global_round",   str(global_round),
            "--dir_round",      str(global_round + 1),  # 1-indexed for directory names
            "--cloud_address",  f"localhost:{CLOUD_PORT}",
        ]

        proxy_proc = subprocess.Popen(proxy_cmd, cwd=project_root, env=proxy_env)
        proxy_client_processes.append((sid, proxy_proc))
        active_processes.append((f"proxy_client_{sid}", proxy_proc))
        time.sleep(0.2)

# ──────────────────────────────────────────────────────────────────────────────
#  4) Wait loops & cleanup
# ──────────────────────────────────────────────────────────────────────────────

def wait_for_leaf_servers_to_finish(global_round: int) -> set[int]:
    """
    Block until each leaf server process exits.
    Return the set of sid's that completed.
    """
    global leaf_server_processes
    logger.info("Waiting for leaf servers to finish their local rounds…")
    completed: set[int] = set()
    attempt = 0
    while len(completed) < NUM_SERVERS and attempt < MAX_SERVER_WAIT_ATTEMPTS:
        time.sleep(1)
        attempt += 1
        # Iterate over a copy in case the original list is modified elsewhere
        for sid, proc in list(leaf_server_processes):
            if sid in completed:
                continue
            if proc.poll() is not None:  # Process has exited
                logger.info(f"Leaf Server {sid} exited with code {proc.returncode}")
                completed.add(sid)
        if attempt % 10 == 0:
            logger.info(f"Still waiting for {NUM_SERVERS - len(completed)} leaf servers… ({attempt}s)")
    if len(completed) < NUM_SERVERS:
        logger.error(f"Timed out waiting for leaf servers after {attempt}s")
    else:
        logger.info("All leaf servers completed successfully!")
    return completed

def wait_for_proxy_clients_to_finish(global_round: int) -> set[int]:
    """
    Block until each proxy client process exits.
    Return the set of sid's that completed.
    """
    global proxy_client_processes
    logger.info("Waiting for proxy clients to complete…")
    completed: set[int] = set()
    attempt = 0
    while len(completed) < NUM_SERVERS and attempt < MAX_PROXY_WAIT_ATTEMPTS:
        time.sleep(1)
        attempt += 1
        for sid, proc in list(proxy_client_processes):
            if sid in completed:
                continue
            if proc.poll() is not None:  # Process has exited
                logger.info(f"Proxy Client {sid} exited with code {proc.returncode}")
                completed.add(sid)
        if attempt % 10 == 0:
            logger.info(f"Still waiting for {NUM_SERVERS - len(completed)} proxy clients… ({attempt}s)")
    if len(completed) < NUM_SERVERS:
        logger.error(f"Timed out waiting for proxy clients after {attempt}s")
    else:
        logger.info("All proxy clients completed successfully!")
    return completed

def wait_for_global_round_to_finish(round_num: int) -> bool:
    """
    Block until "rounds/global/round_{round_num+1}/complete.signal" appears.
    Return True when the signal is found.
    """
    # We use round_num+1 for directory paths (1-indexed)
    dir_round = round_num
    
    # Get the signal path for the current global round (1-indexed for paths)
    round_signal_path = get_global_round_signal_path(dir_round)
    logger.info(f"Waiting for cloud to finish GLOBAL ROUND {dir_round}…")

    attempt = 0
    while attempt < MAX_CLOUD_WAIT_ATTEMPTS:
        time.sleep(1)
        attempt += 1
        if round_signal_path.exists():
            logger.info(f"GLOBAL ROUND {dir_round} completed successfully")
            return True

        # Every 10s, print a reminder
        if attempt % 10 == 0:
            logger.info(f"Still waiting for GLOBAL ROUND {dir_round} to complete... ({attempt}s)")

        # If cloud server died, return False
        if cloud_proc and cloud_proc.poll() is not None:
            logger.error(f"Cloud server exited with code {cloud_proc.returncode}")
            return False

    logger.error(f"Timed out waiting for GLOBAL ROUND {dir_round} after {attempt}s")
    return False

def cleanup_processes():
    """
    Terminate any still‐running subprocesses.
    """
    for name, proc in active_processes:
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    time.sleep(0.5)
    for name, proc in active_processes:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

# ──────────────────────────────────────────────────────────────────────────────
#  Main Orchestration Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global cloud_proc
    """
    Orchestrator main loop. Each global round consists of:
     1. Start the cloud server for this global round (once, at the beginning)
     2. Launch leaf servers
     3. Wait for leaf servers to complete local rounds
     4. Launch Proxy clients to submit models to cloud server
     5. Wait for proxy clients to finish uploading
     6. Wait for cloud to finish aggregating and save global model
    """
    # Track the starting round for each global round
    # Used to tell each leaf server how many rounds it's already done — for logging only.
    total_rounds_so_far = 0

    try:
        # Run each global round
        for global_round in range(GLOBAL_ROUNDS):
            # Start cloud server for this round
            start_cloud_server(global_round)
            print(f"{global_round}/{GLOBAL_ROUNDS-1}")

            # Launch leaf servers
            rounds_to_run = launch_leaf_servers(global_round, total_rounds_so_far)
            total_rounds_so_far += rounds_to_run

            # Wait for leaf servers to finish this round
            completed_servers = wait_for_leaf_servers_to_finish(global_round)

            # Launch proxy clients to submit local models to cloud
            start_proxy_clients(global_round)

            # Wait for proxy clients to finish uploading models
            wait_for_proxy_clients_to_finish(global_round)

            # Wait for cloud to complete aggregation and save global model
            if not wait_for_global_round_to_finish(global_round):
                logger.warning("Did not detect cloud completing the global round")

            # Completed global round
            # Shutdown cloud server for this round
            if cloud_proc and cloud_proc.poll() is None:
                logger.info(f"Shutting down cloud server for GLOBAL ROUND {global_round}")
                cloud_proc.terminate()
                try:
                    cloud_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    cloud_proc.kill()
                cloud_proc = None
                # Brief pause to allow port to free before next cloud start
                time.sleep(1)

        # Create cloud_complete.signal to indicate training is fully complete
        cloud_complete = signals_dir / "cloud_complete.signal"
        cloud_complete.write_text(f"training completed at {datetime.datetime.now()}")
        logger.info("All global rounds completed successfully!")

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, cleaning up...")

    finally:
        # Always clean up processes
        cleanup_processes()
        logger.info("Done.")

if __name__ == "__main__":
    main()
