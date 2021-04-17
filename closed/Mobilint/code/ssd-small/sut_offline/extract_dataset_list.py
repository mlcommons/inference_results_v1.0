import json
import sys
import argparse

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True, help="Name of dataset (COCO/ImageNet)")
    parser.add_argument("--output", required=True, help="Output file name")
    parser.add_argument("--input", required=True, help="Input file name")
    parser.add_argument("--root-path", required=True, help="Path to dataset")
    args = parser.parse_args()
    return args

def strip_coco(dest, src, root_path):
    """ Extract COCO dataset file list """
    print("Stripping COCO")

    files = []

    with open(src) as f_src:
        json_data = json.load(f_src)

        for item in json_data["images"]:
            files.append(root_path + "/" + item["file_name"][:-4] + ".bin")

    with open(dest, "w") as f_dest:
        f_dest.writelines('\n'.join(files))

def strip_imagenet(dest, src, root_path):
    """ Extract ImageNet dataset file list """

    print("Stripping ImageNet")

    files = []

    with open(src) as f_src:
        while True:
            line = f_src.readline()
            
            if not line:
                break

            files.append(root_path + "/" + line.split(" ")[0][:-4] + ".bin")

    with open(dest, "w") as f_dest:
        f_dest.writelines('\n'.join(files))

def main():
    args = get_args()

    if (args.dataset_name == "COCO"):
        strip_coco(args.output, args.input, args.root_path)
    elif (args.dataset_name == "ImageNet"):
        strip_imagenet(args.output, args.input, args.root_path)
    else:
        print("Invalid dataset name")

    print("Done.")

if __name__ == "__main__":
    main()