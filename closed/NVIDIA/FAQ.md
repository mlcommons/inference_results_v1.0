# Common Issues FAQ

**Getting permission issue when container tries to write to local directory**

Update your local directory to allow root to write by `chmod -R 777 <local_dir>`

**I get `useradd: user 'root' already exists` when running `make prebuild`**

Try running `make prebuild` without sudo.

**I get `Got permission denied while trying to connect to the Docker daemon socket` error**

Add yourself to docker group by `sudo usermod -a -G docker $USER`.

**Can I access my local files in the container?**

Local home directory is mounted to `/mnt/home/$USER` in the container. You can also pass in additional flags for Docker
by setting the `DOCKER_ARGS` environment variable, i.e. `make prebuild DOCKER_ARGS="-v my_dir:/my_dir"`.

**How do I install programs like valgrind in the container?**

As usual, `sudo apt-get update`, and then `sudo apt-get install valgrind`.

**I get `nvcc fatal   : Unsupported gpu architecture 'compute_80'` error when running `make build`**

Try to run a clean build. This is due to the previous build having inconsistent CUDA versions.

**Models and data are disappearing, scratch path isn't being linked correctly.**

Make sure that the directory you set `MLPERF_SCRATCH_PATH` to has write permissions for user group `other`. `777` is the
most generic option. This error has also been seen to occur when executing `make` commands in the container with other
users (i.e. using `sudo`), as this will not inherit the `MLPERF_SCRATCH_PATH` environment variable from the default user.
