import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", "-d", default='pacs', help='Dataset')
parser.add_argument("--backbone", "-b", default='resnet50', help='Backbone')
parser.add_argument("--target", "-t", default="sketch", help="Target")
parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
parser.add_argument("--times", "-n", default=1, type=int, help="Repeat times")

args = parser.parse_args()

###############################################################################

if args.dataset == 'pacs':
    source = ["photo", "cartoon", "art_painting", "sketch"]
    input_dir = 'data/pacs'
    config = "PACS/"
elif args.dataset == 'officehome':
    source = ['art', 'clipart', 'product', 'real_world']
    input_dir = 'data/officehome'
    config = "OfficeHome/"

if args.backbone == 'resnet50':
    config += "ResNet50"
elif args.backbone == 'swin_tiny':
    config += "Swin_tiny"
elif args.backbone == 'deit_small':
    config += "Deit_small"

print("config name:", config)

target = args.target
source.remove(target)

output_dir = 'output_FA'

domain_name = target
path = os.path.join(output_dir, config.replace("/", "_"), domain_name)
##############################################################################

for i in range(args.times):
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
              f'python train_FA.py '
              f'--source {source[0]} {source[1]} {source[2]} '
              f'--target {target} '
              f'--input_dir {input_dir} '
              f'--output_dir {output_dir} '
              f'--config {config}')
