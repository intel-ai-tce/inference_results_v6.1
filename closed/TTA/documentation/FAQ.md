# Common Issues FAQ

### I get `Got permission denied while trying to connect to the Docker daemon socket` when launching the container.

Your user is not in the `docker` group. Add it, then restart the daemon:

```
$ sudo usermod -aG docker $USER
$ sudo systemctl restart docker
```

Log out and back in for the group change to take effect. Do not work around this by
running the repo commands with `sudo` &mdash; see the next entry.

### I get permission errors writing to the working directory from inside the container.

This happens when the working directory is not writable by the user the container runs
as. `chmod -R 777` on the repo from outside the container works around it, but the real
fix is the Docker group setup above.

Do not run `make` targets as root. Doing so breaks `MLPERF_SCRATCH_PATH` inheritance and
writes model and dataset files to the wrong place with the wrong ownership.

### How do I access files on the host from inside the container?

The working directory is bind-mounted at `/work`. Pass extra mounts through
`DOCKER_ARGS`:

```
$ make prebuild DOCKER_ARGS="-v /my/data:/my/data"
```

### Models and datasets are disappearing or landing in the wrong directory.

`MLPERF_SCRATCH_PATH` must point at a directory that is writable by the container user.
Confirm it is set inside the container, not just on the host.

### How do I install additional packages inside the container?

The container image is based on Ubuntu 24.04, so `apt` works normally. Anything installed
this way is lost when the container is recreated.

### The run log says `Loadgen built with uncommitted changes!`

LoadGen was built from a modified tree. The submission checker rejects this. Reset the
MLCommons inference checkout to the official v6.1 commit, rebuild LoadGen from
`loadgen/`, and re-run. The check only inspects `loadgen/`, so changes elsewhere in that
tree are fine.

### My throughput differs slightly from the reported result.

`multiple_profiles` lets TensorRT sample different kernel tactics on each engine build,
which moves throughput by roughly &plusmn;0.8% run to run. This is build-time variance, not a
configuration difference. Rebuilding the engine reshuffles it.

### The Server run passed a short sweep but came back INVALID at full length.

Server TTFT is sensitive to run length because the queue keeps building. Always validate
the target QPS with `--min_duration=600000`. A QPS that passes at 180 s can fail the TTFT
constraint at 600 s.

### `make generate_engines` hangs on a `max_num_tokens` assertion.

An engine already on disk was built with a different `max_num_tokens` than the current
config. Delete that engine and rebuild.

### Accuracy is below the ROUGE threshold after quantization.

Check the calibration sequence length before anything else. Calibrating at a length
shorter than the inference input truncates the calibration articles and skews the
activation scales. See [`calibration.md`](calibration.md) and the submission
[`README.md`](../README.md).
