import os
import json
from argparse import ArgumentParser
from PIL import Image

from mmdet.apis import init_detector
from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img_folder', help='Path to folder containing input images'
    )
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    parser.add_argument(
        '--output-dir', default='output', help='Directory to save output images'
    )
    parser.add_argument(
        '--resize-width', type=int, default=1640, help='Width to resize images to'
    )
    parser.add_argument(
        '--resize-height', type=int, default=720, help='Height to resize images to'
    )
    parser.add_argument(
        '--annotations-file', default='annotations.json', help='Output file for annotations'
    )
    args = parser.parse_args()
    return args


def resize_image(image_path, width, height):
    """
    Resize the image to the specified width and height.
    """
    with Image.open(image_path) as img:
        img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
        return img_resized


def create_annotations(img_folder, preds, image_name):
    """
    Create the annotations structure based on the lane detections.
    """
    annotations = {"image_path": os.path.join(img_folder, image_name), "lanes": []}

    # Example: preds contains the detected lane points (assuming a list of lane points)
    # This might vary depending on the output format of your inference
    for lane_id, lane in enumerate(preds):
        lane_points = [(int(pt[0]), int(pt[1])) for pt in lane]  # Assuming each lane is a list of points
        annotations["lanes"].append({f"lane_{lane_id + 1}": lane_points})

    return annotations


def main(args):
    # Initialize the model
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    annotations_list = []

    # Loop through each file in the provided image folder
    for img_name in os.listdir(args.img_folder):
        # Process only common image file types
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(args.img_folder, img_name)
        print(f"Processing: {img_path}")

        # Resize the image
        resized_image = resize_image(img_path, args.resize_width, args.resize_height)
        
        # Save the resized image temporarily (if needed)
        resized_image_path = os.path.join(args.output_dir, "resized_" + img_name)
        resized_image.save(resized_image_path)
        
        # Run inference on the resized image
        src, preds = inference_one_image(model, resized_image_path)

        # Define the output file path using the original image name
        output_path = os.path.join(args.output_dir, img_name)

        # Visualize results and save to the output path
        visualize_lanes(src, preds, save_path=output_path)
        print(f"Saved result to: {output_path}")

        # Create the annotations entry for this image
        annotations = create_annotations(args.img_folder, preds, img_name)
        annotations_list.append(annotations)

        # Optionally, you can delete the resized image after inference (if not needed)
        os.remove(resized_image_path)

    # Save all annotations into a JSON file
    with open(args.annotations_file, 'w') as f:
        json.dump(annotations_list, f, indent=4)
    print(f"Annotations saved to {args.annotations_file}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
