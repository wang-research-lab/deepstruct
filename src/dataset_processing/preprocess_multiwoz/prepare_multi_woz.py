import os
import json
import argparse
import shutil

from extract_examples import extract_dataset_instances


parser = argparse.ArgumentParser(description='Process Multi-WOZ 2.1 Dataset.')
parser.add_argument('--data-dir', type=str, required=True, help='where to save the pre-processed multi-woz data')



def remove_domains_from_dataset(input_path, output_path, rm_domains):
    data = json.load(open(input_path, "r"))
    examples = []
    for x in data["examples"]:
        rm_x = False
        if any([d in x["turn_domain"] for d in rm_domains]):
            continue
        examples.append(x)
    data["examples"] = examples
    json.dump(data, open(output_path, "w"))


def main(args):

    extract_dataset_instances(args.data_dir)
    split_dir = os.path.join(args.data_dir, "splits")

    split_names = ["train", "dev", "test"]
    split_paths = [os.path.join(split_dir, "{0}.json".format(split)) for split in split_names]


    for path, split in zip(split_paths, split_names):
       dir_path = os.path.dirname(path)
       save_path = os.path.join(dir_path, "multi_woz_2.1_{0}_5_domain.json".format(split))
       print("saving split to:", save_path)
       remove_domains_from_dataset(
           input_path=path,
           output_path=save_path,
           rm_domains=["police", "hospital"]
       )
       print("removing: ", path)
       os.remove(path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


