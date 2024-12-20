import os
import argparse
import yaml
import train
import test
import optimize

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help="Select train or test mode")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the configuration YAML file")

    args = parser.parse_args()

    if not args.mode in ['train', 'test', 'opt']:
        raise Exception("Select proper mode")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)  # YAML 파일을 파싱하여 딕셔너리로 변환

    if args.mode == 'train':
        train.main(config, CLASSES, CLASS2IND)
    elif args.mode == 'test':
        test.main(config, IND2CLASS)
    elif args.mode == 'opt':
        optimize.main(config, CLASSES, CLASS2IND)