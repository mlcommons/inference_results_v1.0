# Purpose

It would be nice to have an automated, scenario/benchmark agnostic means of getting timings for a variety of parameters.

Immediately, this is useful for knob-heavy benchmarks like RNN-T, but will also be useful in the future as MLPerf becomes more system-focused (thereby increasing the number of potential parameters needing to be tuned).

# Usage

Requires that the given scenario, benchmark, and system have a present baseline configuration in `configs/` (the typical usage pattern for a new system, is to copy-and-paste an already-tuned system's config entry, and use the autotuner to find a more suitable parameter set).

See

    scripts/autotune/grid.py --help

for arguments.

For "simple" jobs (run all possible combinations of all parameters), it is suggested to use JSON input.

For more complex jobs (where certain combinations shouldn't be executed, special properties are known about the parameters to optimize scheduling, etc), it is suggested to use Python input.
Refer to [Config Schema](#Config-Schema) for additional information.


# Config Schema


## JSON

As seen in `scripts/autotune/example.json`, the expected format is a flat object whose properties are parameter names, and whose values are arrays of non-object, non-array items. Practically, this is bools, numbers, or strings; identical to the python case.


## Python

As seen in `scripts/autotune/example.py`, the expected format is variables with no leading underscores declared at the global namespace which are arrays of primitive, non-dict, non-array type. Practically this is bools, int, float, and str; identical to the JSON case.

Special functions prefixed by `META_` define special hooks into the scheduling process. They are described below:



### META_get_no_rebuild_params()

A function which takes no arguments and returns a list of strings. These strings MUST correspond to variable names in the same Python file (and these strings must also therefore describe valid parameters for the given benchmark/scenario) whose values can be changed by the autotuner without rebuilding the engine.

The scheduler will use the variable names returned by this function to order runs in such a manner to reduce the number of rebuilds (`make generate_engines`) required to fully execute the job.

#### Example

Take the following pathological case:

    audio_batch_size = [128,256,512] # A runtime parameter (doesn't require rebuilding engines)
    gpu_batch_size = [128,256] # A buildtime AND runtime parameter (requires rebuilding engines)

A naive scheduling of these parameters could produce

    audio_batch_size = 128; gpu_batch_size = 128;
    # Rebuild!
    audio_batch_size = 128; gpu_batch_size = 256; 
    # Rebuild!
    audio_batch_size = 256; gpu_batch_size = 128;
    # Rebuild!
    audio_batch_size = 256; gpu_batch_size = 256; 
    # Rebuild!
    audio_batch_size = 512; gpu_batch_size = 128; 
    # Rebuild
    audio_batch_size = 512; gpu_batch_size = 256; 

Requiring 5 rebuilds because a build-time parameter changes at each step

If we defined

    def META_get_no_rebuild_params():
        return ['audio_batch_size']

Our scheduler can produce the following intelligent ordering:

    audio_batch_size = 128; gpu_batch_size = 128;
    audio_batch_size = 256; gpu_batch_size = 128;
    audio_batch_size = 512; gpu_batch_size = 128;
    # Rebuild!
    audio_batch_size = 128; gpu_batch_size = 256;
    audio_batch_size = 256; gpu_batch_size = 256;
    audio_batch_size = 512; gpu_batch_size = 256;

Doing only one rebuild instead! (Potentially reducing our tuning time by an equal order of magnitude for some benchmarks)



### META_is_config_valid()

A callback which takes a "full parameter config dict" (the default config from `configs/benchmark/scenario` updated/overlayed with values from the configuration file), and returns whether or not this configuration is valid.

#### Example

Take the following parameter lists:

    audio_batch_size = [128, 256, 512, 1024]
    dali_pipeline_depth = [1,2,3]
    audio_buffer_num_lines = [128, 256, 512, 1024, 2048, 4096]

And we know that due to the nature of our benchmark, if `audio_batch_size * dali_pipeline_depth` is ever greater than `audio_buffer_num_lines`, our benchmark will crash.
So, the following is true:

    audio_batch_size = 256; dali_pipeline_depth=1; audio_buffer_num_lines = 1024 # Works okay
    audio_batch_size = 512; dali_pipeline_depth=1; audio_buffer_num_lines = 1024 # Works okay
    audio_batch_size = 512; dali_pipeline_depth=2; audio_buffer_num_lines = 1024 # Works okay
    audio_batch_size = 1024; dali_pipeline_depth=2; audio_buffer_num_lines = 1024 # CRASH!

Instead of manually specifying all legal configurations (the exact problem that autotuning solves). What would be nice is if the autotuner could just not run certain configurations based on a predicate. As it turns out, the autotuner script has this interface! It's presents a callback which states if is a configuration is legal before launching work (`generate_engines` and `run_harness`).

So, we can specify:

    def META_is_config_valid(config):
        if config['dali_pipeline_depth'] * config['audio_batch_size'] > config['audio_buffer_num_lines']:
            return False
        return True

And now the failing cases will not be executed.
