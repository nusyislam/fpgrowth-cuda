# Running the parallel experiments on a lab machine

Target machines: the lab GPU server or a Jetson Orin in room 281. The steps
below need only `bash`, `make`, `g++`, `nvcc`, `python3` (standard library)
and coreutils on the lab machine: no virtualenv, pip or internet. Plots are
made back on your own machine.

Replace `USER@LAB` with your login and the machine's host name.

## 1. Package the code and data (on your own machine)

The processed datasets are gitignored, so they travel separately.
`results/bench.csv` is committed and is needed: the benchmark takes its
support levels from it.

```bash
cd ~/fpgrowth
git status --short                       # commit first: git archive ships HEAD only
git archive --format=tar.gz -o /tmp/fpgrowth_src.tgz HEAD
tar czf /tmp/fpgrowth_data.tgz --exclude='data/processed/subsets' data/processed
scp /tmp/fpgrowth_src.tgz /tmp/fpgrowth_data.tgz USER@LAB:
```

(If the lab machine can reach GitHub, `git clone git@github.com:nusyislam/fpgrowth.git`
works instead of the source archive; the data archive is still needed.)

## 2. Unpack and look at the machine (on the lab machine)

```bash
mkdir -p ~/fpgrowth && cd ~/fpgrowth
tar xzf ~/fpgrowth_src.tgz && tar xzf ~/fpgrowth_data.tgz
ls data/processed/                       # chess, connect4, mushroom, online_retail .dat

nproc; free -h                           # cores and memory
g++ --version | head -1
nvcc --version | tail -2                 # Jetson: if missing, export PATH=/usr/local/cuda/bin:$PATH
nvidia-smi                               # server: is anyone else using the GPU right now?
cat /etc/nv_tegra_release 2>/dev/null    # Jetson: L4T / JetPack version
```

Benchmarks are only meaningful if nobody else is running heavy jobs on the
machine. Check `uptime` (load average), `top` and `nvidia-smi` first.

## 3. Build

`ARCH` must match the GPU: `sm_87` for a Jetson Orin. On the server, build
`gpuinfo` first and read its compute capability line. `NVCC_CCBIN` is the
host compiler nvcc uses. The default `g++-13` exists only on the dev
machine, so pass the lab's compiler (plain `g++` usually works; if nvcc
rejects its version, use an older one from `ls /usr/bin/g++-*`).

```bash
make clean
make gpuinfo ARCH=sm_87 NVCC_CCBIN=g++   # Jetson Orin; server: try its ARCH (see below)
./gpuinfo                                # must end "Kernel check on device 0: OK"
```

On the server, if `./gpuinfo` prints e.g. `compute capability  sm_86`, rebuild
with that: `make gpuinfo ARCH=sm_86 NVCC_CCBIN=g++`. A binary built for an
*older* ARCH still runs but JIT-compiles at start-up, which distorts the
timings, so match it exactly.

```bash
make                                     # ./fpgrowth      (serial)
make omp                                 # ./fpgrowth_omp
make cuda ARCH=sm_87 NVCC_CCBIN=g++      # ./fpgrowth_cuda (same ARCH as gpuinfo)
```

## 4. Check correctness on this machine

```bash
make test-par ARCH=sm_87 NVCC_CCBIN=g++  # edge cases, brute force, diff vs serial
```

It must end with `test-par: all passed (./fpgrowth_omp ./fpgrowth_cuda)`. It
takes about 3.5 minutes on the dev machine; expect longer on a Jetson. (`make
test` needs the virtualenv and mlxtend, so skip it on lab machines.)

## 5. Jetson only: power mode

Jetson clocks depend on the power mode, which changes results a lot. Record
it, and use maximum performance if you are allowed to:

```bash
sudo nvpmodel -q                         # current mode (recorded automatically too)
sudo nvpmodel -m 0 && sudo jetson_clocks # max performance (only if allowed)
```

Note in the report which mode was used.

## 6. Run the benchmarks

Run inside `tmux` or `screen` (or with `nohup`), so a dropped SSH connection
does not kill the run. The script is resumable: rerunning it skips every
finished row.

```bash
free -h                                  # want several GB available
tmux new -s bench                        # or: screen -S bench
cd ~/fpgrowth
python3 scripts/run_parallel_bench.py --quick 2>&1 | tee bench_quick.log   # first look, ~10-20 min
python3 scripts/run_parallel_bench.py         2>&1 | tee bench_full.log    # full: 4 levels, 5 reps
```

`--quick` and full runs write to the same CSV (rows from `--quick` have 3 reps
and are kept by the full run). For a clean 5-rep table, delete
`results/parallel_$(hostname -s).csv` before the full run.

What it does: one job at a time. Each run is capped at 120 s and 4 GB
resident memory (40% of RAM on smaller machines). It checks available
memory before every support level and refuses to start below
min(6 GB, half of RAM). A config that times out is not run at harder levels.
On the dev machine the full run took about 35 minutes. A Jetson's CPU is
several times slower, so plan for 2+ hours, or run fewer datasets
(`python3 scripts/run_parallel_bench.py chess online_retail`).

Outputs (named by host name, so machines never overwrite each other):

- `results/parallel_<host>.csv`: one row per dataset, support level and config (median times)
- `results/gpu_threshold_<host>.csv`: the `--gpu-min-nodes` sweep
- `results/gpu_info_<host>.txt`: CPU, cores, RAM, OS, compilers, CUDA, `gpuinfo` output, Jetson power mode

Then the kernel micro-benchmark (seconds to a minute). It times only the
GPU counting kernel against the CPU on the initial tree:

```bash
python3 scripts/kernel_micro.py 2>&1 | tee kernel_micro.log   # -> results/kernel_micro_<host>.csv
```

## 7. Copy the results back and plot (on your own machine)

```bash
cd ~/fpgrowth
scp 'USER@LAB:fpgrowth/results/parallel_*.csv' 'USER@LAB:fpgrowth/results/gpu_threshold_*.csv' \
    'USER@LAB:fpgrowth/results/kernel_micro_*.csv' 'USER@LAB:fpgrowth/results/gpu_info_*.txt' results/
make figures                             # adds omp_speedup_<host>, impl_compare_<host>,
                                         # gpu_threshold_<host>, kernel_micro_<host>
git add results/ && git commit -m "Parallel benchmark results from <host>"
```

## 8. Clean up the lab machine

```bash
rm -rf ~/fpgrowth ~/fpgrowth_src.tgz ~/fpgrowth_data.tgz
sudo nvpmodel -m <previous mode>         # Jetson, if you changed it
```
