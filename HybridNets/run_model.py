import time
import torch
from torch.backends import cudnn
from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
from torchvision import transforms
import argparse
from utils.constants import *
from collections import OrderedDict
from torch.nn import functional as F

def parse_args():

    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('--source', type=str, default='data/input', help='The demo image folder')
    parser.add_argument('--output', type=str, default='data/output', help='Output folder')
    parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
    parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
    parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
    parser.add_argument('--imshow', type=boolean_string, default=False, help="Show result onscreen (unusable on colab, jupyter...)")
    parser.add_argument('--imwrite', type=boolean_string, default=False, help="Write result to output folder")
    parser.add_argument('--show_det', type=boolean_string, default=False, help="Output detection result exclusively")
    parser.add_argument('--show_seg', type=boolean_string, default=True, help="Output segmentation result exclusively")
    parser.add_argument('--cuda', type=boolean_string, default=True)
    parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
    parser.add_argument('--speed_test', type=boolean_string, default=False,
                        help='Measure inference latency')
    args = parser.parse_args()
    return args

def main(args):
    params = Params(f'projects/{args.project}.yml')
    color_list_seg = {}
    for seg_class in params.seg_list:
        # edit your color here if you wanna fix to your liking
        color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))
    
    weight = args.load_weights    
    input_imgs = []
    shapes = []
    det_only_imgs = []

    compound_coef = args.compound_coef
    source = args.source

    anchors_ratios = params.anchors_ratios
    anchors_scales = params.anchors_scales

    threshold = args.conf_thresh
    iou_threshold = args.iou_thresh

    use_cuda = args.cuda
    use_float16 = args.float16
    cudnn.fastest = True
    cudnn.benchmark = True

    imshow = args.imshow
    imwrite = args.imwrite
    show_det = args.show_det
    show_seg = args.show_seg

    obj_list = params.obj_list
    seg_list = params.seg_list
    color_list = standard_to_bgr(STANDARD_COLORS)

    resized_shape = params.model['image_size']
    if isinstance(resized_shape, list):
        resized_shape = max(resized_shape)

    normalize = transforms.Normalize( mean=params.mean, std=params.std)

    transform = transforms.Compose([transforms.ToTensor(), normalize,])
    
    weight = torch.load(weight, map_location='cuda' if use_cuda else 'cpu')
    weight_last_layer_seg = weight['segmentation_head.0.weight']
    if weight_last_layer_seg.size(0) == 1:
        seg_mode = BINARY_MODE
    else:
        if params.seg_multilabel:
            seg_mode = MULTILABEL_MODE
        else:
            seg_mode = MULTICLASS_MODE
    print('Loading Hybridnets model...')
    model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                           scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=args.backbone,
                           seg_mode=seg_mode)
    model.load_state_dict(weight)

    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    ori_imgs = []
    shapes = []
    input_imgs = []
    det_only_imgs = []
    
    for subdir, _, files in os.walk('data/input'):
        if files:
            parent_dir = os.path.basename(os.path.dirname(subdir))
            current_dir = os.path.basename(subdir)
            if parent_dir != 'input':
                print(f'Processing hybridnets files in {parent_dir}/{current_dir}')
                output_path = os.path.join('data/output', f'{parent_dir}/{current_dir}')
            else:
                print(f'Processing hybridnets files in {current_dir}')
                output_path = os.path.join('data/output', current_dir)
            os.makedirs(output_path, exist_ok=True)
            for file in files:
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
                ori_imgs.append(img)
                h0, w0 = img.shape[:2]
                r = resized_shape / max(h0, w0)
                input_img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
                h, w = input_img.shape[:2]
                (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True,
                                                       scaleup=False)
                
                input_imgs.append(input_img)
                shapes.append(((h0, w0), ((h / h0, w / w0), pad)))
                if use_cuda:
                    x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
                else:
                    x = torch.stack([transform(fi) for fi in input_imgs], 0)
                x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)

                with torch.no_grad():
                    features, regression, classification, anchors, seg = model(x)
                    # in case of MULTILABEL_MODE, each segmentation class gets their own inference image
                    seg_mask_list = []
                    # (B, C, W, H) -> (B, W, H)
                    if seg_mode == BINARY_MODE:
                        seg_mask = torch.where(seg >= 0, 1, 0)
                        # print(torch.count_nonzero(seg_mask))
                        seg_mask.squeeze_(1)
                        seg_mask_list.append(seg_mask)
                    elif seg_mode == MULTICLASS_MODE:
                        _, seg_mask = torch.max(seg, 1)
                        seg_mask_list.append(seg_mask)
                    else:
                        seg_mask_list = [torch.where(torch.sigmoid(seg)[:, i, ...] >= 0.5, 1, 0) for i in range(seg.size(1))]
                        # but remove background class from the list
                        seg_mask_list.pop(0)
                    regressBoxes = BBoxTransform()
                    clipBoxes = ClipBoxes()
                    out = postprocess(x,
                                    anchors, regression, classification,
                                    regressBoxes, clipBoxes,
                                    threshold, iou_threshold)

                                    #(B, W, H) -> (W, H)
                    for i in range(seg.size(0)):
                        #   print(i)
                        for seg_class_index, seg_mask in enumerate(seg_mask_list):
                            seg_mask_ = seg_mask[i].squeeze().cpu().numpy()
                            pad_h = int(shapes[i][1][1][1])
                            pad_w = int(shapes[i][1][1][0])
                            seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
                            seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)
                            color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
                            for index, seg_class in enumerate(params.seg_list):
                                    color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
                            color_seg = color_seg[..., ::-1]  # RGB -> BGR
                            # cv2.imwrite(f'{output_path}/{file}', color_seg)
                
                            color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background
                            # prepare to show det on 2 different imgs
                            # (with and without seg) -> (full and det_only)
                            # det_only_imgs.append(ori_imgs[i].copy())
                            seg_img = ori_imgs[i].copy() if seg_mode == MULTILABEL_MODE else ori_imgs[i]  # do not work on original images if MULTILABEL_MODE
                            seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
                            seg_img = seg_img.astype(np.uint8)
                            seg_filename = f'{output_path}/{file}_{params.seg_list[seg_class_index]}' if seg_mode == MULTILABEL_MODE else \
                                           f'{output_path}/{file}'
                            
                            cv2.imwrite(seg_filename, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
                    
                    # for i in range(len(ori_imgs)):
                    #     out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])
                    #     for j in range(len(out[i]['rois'])):
                    #         x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
                    #         obj = obj_list[out[i]['class_ids'][j]]
                    #         score = float(out[i]['scores'][j])
                    #         plot_one_box(ori_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                    #                     color=color_list[get_index_label(obj, obj_list)])
                            
                    #         plot_one_box(det_only_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                    #                     color=color_list[get_index_label(obj, obj_list)])
                    #         fname = f'{output_path}/{file}'
                    #         cv2.imwrite(fname,cv2.cvtColor(det_only_imgs[i], cv2.COLOR_RGB2BGR))
                    
                input_imgs.clear()
                shapes.clear()
                ori_imgs.clear()
                det_only_imgs.clear()


if __name__ == '__main__':
    args = parse_args()
    main(args)