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

### META_search_callback()

A _stateless_ callback which returns a sequence of parameter combinations to be executed. As an argument, takes  a list of past runs, which is of type: `List[Tuple[Dict, Dict]]` or more verbosely: `List[Tuple[ParamDict, PerfStatDict]]`. Returns `ParamDict` while searching and returns `None` to signal search has concluded.

This should be used when the default behavior of exhaustively iterating through all parameter combinations is not desired.


#### Helpers

Refer to `library.py` which provides `QPSBisect` a stateless binary search for QPS. The helper can be imported with `from library import QPSBisect` (note that package lookup is relative to `grid.py`, not the configuration py file, so no fancy relative pathing or `sys.path.insert(foo)` needed.)

#### Example: Binary Search

Suppose we want to perform a binary search to find an optimal QPS value given default parameters in the QPS range: `(500, 2500)` in step-sizes of 100. We can use `QPSBisect` to make the following

    from library import QPSBisect

    def META_search_callback(past_runs):
        b = QPSBisect(lb=500, ub=2500, ss=100, past_runs=past_runs)
        # Will return a new cross-term until solution found. Then returns None
        return b.get_next_cross()

    # We know QPS is a runtime-only parameter, so we don't need to rebuild in between runs:
    def META_get_no_rebuild_params():
        return ["server_target_qps"]

#### Example: Exaustive Batch Size + Binary Search QPS

But now, let's say we want to do this binary search for a set of batch sizes. We can compose QPS bisecting with a linear batch size search. (Note that the extra verbosity is needed for state-less-ness):

    from library import QPSBisect

    # Define a helper which partially fills out QPSBisect
    def bisector(past_runs):
        return QPSBisect(lb=500, ub=2500, ss=100, past_runs=past_runs)

    # Leading underscore for module-private visibility
    _batch_sizes_to_test = [256, 512, 1024]
    server_target_qps=[]
    def META_get_no_rebuild_params():
        return ['server_target_qps']

    def META_search_callback(past_runs):
        if len(past_runs) == 0:
            # First time executing
            c = bisector(None).get_next_cross()
            c['gpu_batch_size'] = _batch_sizes_to_test[0]
        else:
            # Check what the most recent batch size was:
            last_bs = past_runs[-1][0]['gpu_batch_size']
            # Get all past runs with our current BS to not trip-up QPSBisect
            filtered_past = [p for p in past_runs if p[0]['gpu_batch_size'] == last_bs]
            # Test if we've exhausted this BS:
            c = bisector(filtered_past).get_next_cross()
            if c is None:
                # Use next batch size:
                next_bs_idx = _batch_sizes_to_test.index(last_bs) + 1
                if next_bs_idx >= len(_batch_sizes_to_test):
                    # No next bs to do
                    c = None
                else:
                    c = bisector(None).get_next_cross()
                    c['gpu_batch_size'] = _batch_sizes_to_test[next_bs_idx]
            else:
                # We're still searching through the current BS
                c['gpu_batch_size'] = last_bs
        return c


