# Shared by the Slurm scripts: sourced, never run directly.
# Expects to be sourced from the `final-assignment` folder.

REQUIRED_KEYS="PARTITION GPUS CPUS_PER_GPU CONTAINER_IMAGE CONTAINER SHARED_DIR LOG_DIR"
TRACKS="segmentation consistency realtime generative"

# Load slurm/servers/$SERVER.env (default: snellius) and check that nothing is missing.
load_server() {
    SERVER=${SERVER:-snellius}
    local config="slurm/servers/$SERVER.env"
    if [ ! -f "$config" ]; then
        echo "No server config $config. Available: $(ls slurm/servers | sed 's/\.env$//' | tr '\n' ' ')" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$config"
    local missing="" key
    for key in $REQUIRED_KEYS; do
        [ -n "${!key:-}" ] || missing="$missing $key"
    done
    if [ -n "$missing" ]; then
        echo "$config leaves these empty:$missing" >&2
        exit 1
    fi
    export SERVER
}

# Stop with a clear message if $1 is not a track.
check_track() {
    case " $TRACKS " in
        *" $1 "*) ;;
        *) echo "Unknown track '$1'. Use one of: $TRACKS" >&2; exit 1 ;;
    esac
}

# Short report at the top of every job log.
print_job_info() {
    echo "===== JOB ====="
    echo "Job: ${SLURM_JOB_NAME:-} ${SLURM_JOB_ID:-}  Node: ${SLURM_NODELIST:-$(hostname)}  Server: $SERVER"
    echo "Command: $*"
    git log -1 --format='Commit: %h %s' 2>/dev/null || true
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo "===== OUTPUT ====="
}

# Run a command inside the container, with your .env (W&B key) and the shared data visible.
# With DRY_RUN=1 the command is printed instead of run.
in_container() {
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf '%q ' "$@"; echo
        return 0
    fi
    local binds=() path
    for path in $BIND_PATHS; do binds+=(--bind "$path"); done
    srun apptainer exec --nv "${binds[@]}" --env-file .env "$CONTAINER" "$@"
}
