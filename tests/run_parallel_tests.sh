#!/usr/bin/env bash
# Correctness of the parallel binaries (make test-par).
#
#   tests/run_parallel_tests.sh
#
# For ./fpgrowth_omp, and ./fpgrowth_cuda if it was built and a GPU is usable:
#   1. the edge cases (tests/run_edge_cases.sh) at several thread counts;
#   2. the brute-force differential test (tests/brute_force.py);
#   3. on every processed dataset at 2 support levels, the sorted output must
#      be byte-identical to ./fpgrowth's, the --unsorted output must have the
#      same lines (compared after sorting), and the itemset / conditional-tree
#      counters on stderr must match the serial run.
# fpgrowth_cuda also runs with --gpu-min-nodes 0 (every conditional tree with
# two or more items is counted on the GPU) or 64 on the larger datasets, so
# the offload path is exercised far more than with the default threshold.
#
# Needs only bash, coreutils and python3 (standard library).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export LC_ALL=C   # byte order for sort, as in the binaries' output

fail=0
step() { echo; echo "== $*"; }
check() {  # check <description> <command...>: run, record failure
    local what=$1; shift
    if "$@"; then :; else echo "FAIL: $what"; fail=$((fail + 1)); fi
}

[ -x ./fpgrowth ] && [ -x ./fpgrowth_omp ] || { echo "build ./fpgrowth and ./fpgrowth_omp first (make test-par)"; exit 1; }

BINS=(./fpgrowth_omp)
if [ -x ./fpgrowth_cuda ]; then
    if gpu_err=$(./fpgrowth_cuda data/examples/han.dat 2 2>&1 >/dev/null); then
        BINS+=(./fpgrowth_cuda)
    else
        echo "SKIP fpgrowth_cuda: it does not run here ($(echo "$gpu_err" | tail -1))"
    fi
else
    echo "SKIP fpgrowth_cuda: not built (no nvcc?)"
fi

# --- 1. edge cases -----------------------------------------------------------
for bin in "${BINS[@]}"; do
    step "edge cases: $bin"
    check "edge cases $bin" bash tests/run_edge_cases.sh "$bin"
    check "edge cases $bin --threads 1" bash tests/run_edge_cases.sh "$bin" --threads 1
    check "edge cases $bin --threads 3" bash tests/run_edge_cases.sh "$bin" --threads 3
    if [ "$bin" = ./fpgrowth_cuda ]; then
        check "edge cases $bin --gpu-min-nodes 0" bash tests/run_edge_cases.sh "$bin" --gpu-min-nodes 0
    fi
done

# --- 2. brute force ----------------------------------------------------------
step "brute force: ./fpgrowth_omp"
check "brute force omp t4" "$PY" tests/brute_force.py --binary ./fpgrowth_omp --args "--threads 4" --trials 300 --seed 11
check "brute force omp t1" "$PY" tests/brute_force.py --binary ./fpgrowth_omp --args "--threads 1" --trials 100 --seed 12
if [[ " ${BINS[*]} " == *" ./fpgrowth_cuda "* ]]; then
    # Fewer trials: every run pays the CUDA context start-up (~0.15 s).
    step "brute force: ./fpgrowth_cuda"
    check "brute force cuda min0 t4" "$PY" tests/brute_force.py --binary ./fpgrowth_cuda \
        --args "--threads 4 --gpu-min-nodes 0" --trials 100 --seed 13
    check "brute force cuda default" "$PY" tests/brute_force.py --binary ./fpgrowth_cuda --trials 40 --seed 14
fi

# --- 3. real datasets vs the serial binary -----------------------------------
# name  support  support  cuda --gpu-min-nodes for the extra offload run
LEVELS="
chess          0.7     0.5      0
mushroom       0.1     0.05     0
connect4       0.8     0.7      0
online_retail  0.005   0.0025   64
"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
counters() {  # serial-binary counters only (not e.g. gpu_tree_nodes)
    tr ' ' '\n' <"$1" | grep -E '^(itemsets|cond_trees|cond_nodes|single_paths|frequent_items|tree_nodes)=' | tr '\n' ' '
}

compare() {  # compare <label> <binary> <dat> <support> [args...]
    local label=$1 bin=$2 dat=$3 sup=$4; shift 4
    "$bin" "$dat" "$sup" "$tmp/par.txt" "$@" 2>"$tmp/par.err" || { echo "FAIL $label: exit $?"; cat "$tmp/par.err"; return 1; }
    if ! cmp -s "$tmp/serial.txt" "$tmp/par.txt"; then
        echo "FAIL $label: sorted output differs from ./fpgrowth"; return 1
    fi
    if [ "$(counters "$tmp/serial.err")" != "$(counters "$tmp/par.err")" ]; then
        echo "FAIL $label: counters differ"; echo "  serial: $(counters "$tmp/serial.err")"
        echo "  this:   $(counters "$tmp/par.err")"; return 1
    fi
    "$bin" "$dat" "$sup" "$tmp/par_u.txt" --unsorted "$@" 2>/dev/null || { echo "FAIL $label --unsorted: exit $?"; return 1; }
    sort "$tmp/par_u.txt" >"$tmp/par_us.txt"
    if ! cmp -s "$tmp/serial_us.txt" "$tmp/par_us.txt"; then
        echo "FAIL $label --unsorted: lines differ from ./fpgrowth --unsorted"; return 1
    fi
    echo "PASS $label ($(grep -oE 'itemsets=[0-9]+' "$tmp/par.err"))"
}

step "datasets vs ./fpgrowth"
found=0
while read -r name s1 s2 minnodes; do
    [ -z "$name" ] && continue
    dat=data/processed/$name.dat
    if [ ! -f "$dat" ]; then echo "SKIP $name: $dat missing (make data)"; continue; fi
    found=1
    for sup in $s1 $s2; do
        ./fpgrowth "$dat" "$sup" "$tmp/serial.txt" 2>"$tmp/serial.err"
        ./fpgrowth "$dat" "$sup" "$tmp/serial_u.txt" --unsorted 2>/dev/null
        sort "$tmp/serial_u.txt" >"$tmp/serial_us.txt"
        check "$name $sup omp" compare "$name $sup omp (all threads)" ./fpgrowth_omp "$dat" "$sup"
        check "$name $sup omp t2" compare "$name $sup omp --threads 2" ./fpgrowth_omp "$dat" "$sup" --threads 2
        if [[ " ${BINS[*]} " == *" ./fpgrowth_cuda "* ]]; then
            check "$name $sup cuda" compare "$name $sup cuda (all threads)" ./fpgrowth_cuda "$dat" "$sup"
            check "$name $sup cuda min" compare "$name $sup cuda --gpu-min-nodes $minnodes" \
                ./fpgrowth_cuda "$dat" "$sup" --gpu-min-nodes "$minnodes"
        fi
    done
done <<EOF
$LEVELS
EOF
[ $found -eq 1 ] || echo "SKIP datasets: none processed (make data, or copy data/processed/)"

echo
if [ $fail -eq 0 ]; then
    echo "test-par: all passed (${BINS[*]})"
else
    echo "test-par: $fail check(s) FAILED"
    exit 1
fi
