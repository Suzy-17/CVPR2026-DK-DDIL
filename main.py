import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import argparse
from trainer import train

def main(): 
    args = setup_parser().parse_args()
    # args.config = "exps/skin.json"
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json
    # if args['seq'] == 'racp':
    #     args['task_name'] = ["Real_World", "Art", "Clipart", "Product"]
    #     args['prefix'] = "racp_"+args['prefix']
    # elif args['seq'] == 'prac':
    #     args['task_name'] = ["Product", "Real_World", "Art", "Clipart"]
    #     args['prefix'] = args['seq']+"_"+args['prefix']
    # elif args['seq'] == 'cpra':
    #     args['task_name'] = ["Clipart", "Product", "Real_World", "Art"]
    #     args['prefix'] = args['seq']+"_"+args['prefix']
    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorithms.')
    parser.add_argument('--config', type=str, default='exps/skin_ours_class_select.json',
                        help='Json file of settings.')
    # parser.add_argument('--seq', type=str, default='racp',
    #                     help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
