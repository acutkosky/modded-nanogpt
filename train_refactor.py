# A more configurable script with light modifications
# to better fit SCC job submitting.

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention

# --- New libraries ---
import argparse
import ast
import wandb
from dataclasses import asdict
# move optimizers to a different folder for convenient configurations
from optimizers import Muon, Mango, SFMuon
from modeling import GPT

# -----------------------------------------------------------------------------
# Additional argparser to interface with cmd and parallel submit

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true"):
        return True
    elif v.lower() in ("false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def str2tuple(v):
    assert isinstance(v, str)
    res = [ast.literal_eval(e.strip()) for e in v.split(",")]
    return tuple(res)

def parse_args():
    parser = argparse.ArgumentParser(description="Additional cmd args.")
    # basics
    parser.add_argument("--random_seed", type=int, default=42, help="Fix a random seed")
    parser.add_argument("--optimizer", type=str, default="muon", help="Optimizer name")
    # logging
    parser.add_argument("--log_folder", type=str, default="", help="Log subfolder name")
    parser.add_argument("--run_name", type=str, default="", help="Name your run")
    parser.add_argument("--wandb_project", type=str, default="nanogpt_speedrun", help="Log to wandb project name")
    # some auxiliary args
    parser.add_argument("--compile_only", type=str2bool, default=False, help="Turn on to break after compiling.")
    parser.add_argument("--advanced_log", type=str2bool, default=False, help="Turn on to log advanced info")
    # optimizer-specific: Mango
    parser.add_argument("--mango_mat_lr", type=float, default=0.05, help="Mango-mat learning rate")
    parser.add_argument("--mango_mat_beta1", type=str2tuple, default="0.85,0.95,300", help="Mango-mat beta1")
    parser.add_argument("--mango_mat_beta2", type=str2tuple, default="0,0,300", help="Mango-mat beta2")
    parser.add_argument("--mango_mat_nesterov", type=str2bool, default=True, help="Mango-mat nesterov momentum")
    parser.add_argument("--mango_mat_backend", type=str, default="newtonschulz5", help="Mango_mat normalize backend")
    parser.add_argument("--mango_mat_backend_args", type=str, default="steps=5,scale_dim=True", help="Mango_mat backend extra args")
    parser.add_argument("--mango_mat_scale_rms", type=str2bool, default=False, help="Mango_mat normalize update by rms norm")
    parser.add_argument("--mango_mat_grafting", type=str2bool, default=False, help="Mango_mat use grafting")
    parser.add_argument("--mango_mat_eps", type=float, default=1e-8, help="Mango_mat eps")
    parser.add_argument("--mango_mat_use_cond", type=str2bool, default=False, help="Mango_mat turn on conditioning")
    parser.add_argument("--mango_mat_laprop", type=str2bool, default=False, help="Mango_mat use laprop pre-conditioning")
    parser.add_argument("--mango_mat_precond_power", type=float, default=0.0, help="Mango_mat preconditioning power")
    parser.add_argument("--mango_mat_postcond_power", type=float, default=0.0, help="Mango_mat postconditioning power")
    # optimizer-specific: SFMuon
    parser.add_argument("--sfmuon_lr", type=float, default=0.05)
    parser.add_argument("--sfmuon_momentum", type=str2tuple, default="0.85,0.95,300")
    parser.add_argument("--sfmuon_nesterov_beta", type=str2tuple, default="0.85,0.95,300")
    return parser.parse_args()

cmd_args = parse_args()

def parse_backend_args(args: str) -> dict:
    res = {}
    for arg in args.split(","):
        k, v = arg.strip().split("=")
        res[k] = ast.literal_eval(v)
    return res

# -----------------------------------------------------------------------------
# Reproducibility: Set the random seed (adjust base_seed as desired)
import random
import numpy as np

base_seed = cmd_args.random_seed
rank = int(os.environ["RANK"])
seed = base_seed + rank

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# init main

@dataclass
class Hyperparameters:
    # data
    data_dir: str = "data"
    train_files = "fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 1770 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
    
args = Hyperparameters(
    data_dir="/projectnb/aclab/datasets",
)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
# assert world_size == 8 # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = str(uuid.uuid4())
    run_name = f"{cmd_args.run_name}_{run_id[:4]}"
    log_dir = f"logs/{cmd_args.log_folder}"
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"{run_name}.txt")
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# additionally print the cmd_args
print0(vars(cmd_args))
print0("="*100)
# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

########################################
#    Construct model and optimizer     #
########################################

model: torch.nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
for m in model.modules():
    if isinstance(m, torch.nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
if cmd_args.optimizer == "muon":
    adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2]
elif cmd_args.optimizer == "mango":
    adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    # optimizer1 = Mango(adam_params, beta1=0.8, beta2=0.95, nesterov=False,
    #                    backend=None, scale_rms=False, eps=1e-10, laprop=False,
    #                    precond_power=0.5, postcond_power=0.0)
    # NOTE: the current implementation of Mango doesn't recover Adam: momentum doesn't have (1-beta), and there's no debiasing
    # so, for now we first keep Adam as the backup, and only test on muon vs mango.
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Mango(hidden_matrix_params, 
                       lr=cmd_args.mango_mat_lr, 
                       beta1=cmd_args.mango_mat_beta1[1],   # change to tuple 
                       beta2=cmd_args.mango_mat_beta2[1], 
                       nesterov=cmd_args.mango_mat_nesterov,
                       backend=cmd_args.mango_mat_backend,
                       scale_rms=cmd_args.mango_mat_scale_rms, 
                       grafting=cmd_args.mango_mat_grafting,
                       eps=cmd_args.mango_mat_eps, 
                       use_cond=cmd_args.mango_mat_use_cond,
                       laprop=cmd_args.mango_mat_laprop,
                       precond_power=cmd_args.mango_mat_precond_power, 
                       postcond_power=cmd_args.mango_mat_postcond_power,
                       **parse_backend_args(cmd_args.mango_mat_backend_args))
    optimizers = [optimizer1, optimizer2]
elif cmd_args.optimizer == "sfmuon":
    adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = SFMuon(hidden_matrix_params, 
                        lr=cmd_args.sfmuon_lr, 
                        momentum=cmd_args.sfmuon_momentum[1],
                        nesterov_beta=cmd_args.sfmuon_nesterov_beta[1],
                        rank=rank, 
                        world_size=world_size)
    optimizers = [optimizer1, optimizer2]
else:
    raise ValueError(f"optimizer='{cmd_args.optimizer}' not implemented.")
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

model: torch.nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

# Compile on a new node for test purpose
if cmd_args.compile_only:
    raise KeyboardInterrupt

########################################
#           Logging to WandB           #
########################################

if master_process:
    wandb.init(
        project=cmd_args.wandb_project, 
        name=run_name,
        id=run_id,
        resume="never",
    )
    wandb.config.update(
        {**asdict(args), **vars(cmd_args)}
    )

########################################
#        Training and validation       #
########################################

def warmup_momentum(step, start, end, warmup):
    frac = min(step / warmup, 1)
    return (1 - frac) * start + frac * end

# Simulate parallel training on a singl GPU
simulate_world_size = 8
assert simulate_world_size % world_size == 0
meta_batch_size = simulate_world_size // world_size

train_loader = distributed_data_generator(
    os.path.join(args.data_dir, args.train_files), world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(
            os.path.join(args.data_dir, args.val_files), val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        # add wandb_logging
        if master_process:
            wandb.log({"val_loss": val_loss}, step=step)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    # Simulate parallel training:
    train_loss = 0
    for _ in range(meta_batch_size):
        inputs, targets = next(train_loader)
        loss = model(inputs, targets, get_window_size_blocks(step)) / meta_batch_size
        loss.backward()
        train_loss += loss.detach() / args.train_seq_len    # NOTE: args.train_seq_len is the local batch size per gpu
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    # muon-specific
    if cmd_args.optimizer == "muon":
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # mango-specific
    if cmd_args.optimizer == "mango":
        for group in optimizer2.param_groups:
            group["beta1"] = warmup_momentum(step, *cmd_args.mango_mat_beta1)
            group["beta2"] = warmup_momentum(step, *cmd_args.mango_mat_beta2)
    # sfmuon-specific
    if cmd_args.optimizer == "sfmuon":
        for group in optimizer2.param_groups:
            momentum_start, momentum_end, momentum_warmup = cmd_args.sfmuon_momentum
            nesterov_start, nesterov_end, nesterov_warmup = cmd_args.sfmuon_nesterov_beta
            frac1 = min(step / momentum_warmup, 1)
            frac2 = min(step / nesterov_warmup, 1)
            group["momentum"] = (1 - frac1) * momentum_start + frac1 * momentum_end
            group["nesterov_beta"] = (1 - frac2) * nesterov_start + frac2 * nesterov_end
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_loss:{train_loss} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
    if master_process:
        # NOTE: this train_loss is the local loss on the master node, not averaged over all nodes.
        wandb.log({"loss": train_loss}, step=step)
        # Visualizing: 
        if cmd_args.advanced_log:
            opt_metrics = {}
            id_to_name = {id(param): name for name, param in model.named_parameters()}
            for param, state in optimizer2.state.items():
                name = id_to_name.get(id(param))
                param_logs = state.get("logs")
                if name is not None and param_logs is not None:
                    opt_metrics.update({
                        f"{k}/{name}": v for k, v in param_logs.items()
                    })
            wandb.log(opt_metrics, step=step)
    # null the gradients
    model.zero_grad(set_to_none=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group() 