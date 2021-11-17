import os
import os.path as osp
import numpy as np
import argparse
from collections import defaultdict
import json
from decimal import Decimal


def write_now(row, colwidth=14):
    sep = "  "

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.2f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    return sep.join([format_val(x) for x in row]) + "\n"


def to_decimal(a):
    return Decimal(a).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", default=None, type=str)
    args = parser.parse_args()

    directory = args.directory
    dataset = directory.split('_')[0]
    directory = osp.join('output_FA', directory)
    domains = os.listdir(directory)
    domains.sort()

    collect_path = osp.join(directory, 'collect_results.txt')
    fout = open(collect_path, 'w')
    fout.write(write_now([dataset] + domains + ['averge']))

    row_list = []
    mean_list = []
    row_list.append('FACT')
    for domain in domains:
        sub_dir = osp.join(directory, domain)
        print(sub_dir)
        trials = os.listdir(sub_dir)
        assert len(trials) == 5, "not complete!"
        results = []
        for trial in trials:
            file_name = osp.join(sub_dir, trial, 'best_acc.json')
            with open(file_name, 'r', encoding='utf-8') as f:
                load_data = json.load(f)
                best_acc = load_data['acc_val_best_test']
                results.append(best_acc * 100)
        
        mean = np.mean(results)
        std = np.std(results, ddof=1)
        mean = to_decimal(mean)
        std = to_decimal(std)

        row_list.append("& {}±{}".format(mean, std))
        mean_list.append(mean)

    average_acc = sum(mean_list) / len(mean_list)
    average_acc = average_acc.quantize(
        Decimal("0.01"), rounding="ROUND_HALF_UP"
    )
    row_list.append("& {}".format(average_acc))
    fout.write(write_now(row_list))

    fout.close()


                

    

