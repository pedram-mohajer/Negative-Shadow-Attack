from argparse import ArgumentParser

from mmdet.apis import init_detector

from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('folder', help='input folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='result.png', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    for subdir, _, files in os.walk('data/input'):
        if files:
            parent_dir = os.path.basename(os.path.dirname(subdir))
            current_dir = os.path.basename(subdir)
            if parent_dir != 'input':
                print(f'Processing clrernet files in {parent_dir}/{current_dir}')
                output_path = os.path.join('data/output', f'{parent_dir}/{current_dir}')
            else:
                print(f'Processing clrernet files in {current_dir}')
                output_path = os.path.join('data/output', current_dir)
            
            os.makedirs(output_path, exist_ok=True)

            for file in files:
                img_path = os.path.join(subdir, file)
                src, preds = inference_one_image(model, img_path)
                # show the results
                dst = visualize_lanes(src, preds, save_path=os.path.join(output_path, file))


if __name__ == '__main__':
    args = parse_args()
    main(args)
