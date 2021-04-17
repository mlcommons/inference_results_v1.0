
import sys
import json
import collections

def configParser( filename ):
    try:
        with open(filename, 'rb') as fid:
            config_data = json.load(fid)
            cutoffs = config_data["cutoffs"]
            batch_sizes = config_data["batch_sizes"]
            instances = config_data["instances"]
            cores = config_data["cores_per_bucket_instances"]
    except Exception as msg:
        print("[ERROR] Unable to read server config file {} : {}".format( filename, msg))
        sys.exit(1)
    
    buckets = {}
    for j, cutoff in enumerate(cutoffs):
        batch_size = batch_sizes[j]
        num_instance = instances[j]
        cores_per_instance = cores[j]

        buckets[ cutoff ] = {"batch_size": batch_size, "instances": num_instance, "cpus_per_instance": cores_per_instance}

    return buckets
