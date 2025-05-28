# hybridnet.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
import torch
from   torch.backends import cudnn
from   hybrid.backbone import HybridNetsBackbone
import cv2
import numpy as np
from   glob import glob
from   hybrid.utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
       boolean_string, Params
from   hybrid.utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
from   torchvision import transforms
import argparse
from   hybrid.utils.constants import *
from   collections import OrderedDict
from   torch.nn import functional as F



from typing import List, Tuple

def run_hybridnets(bright_pixels_driver: List[Tuple[int, int]], shadow_id: int) -> Tuple[bool, float]:


#def run_hybridnets(img_path):

    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('--source', type=str, default='demo/image', help='The demo image folder')
    parser.add_argument('--output', type=str, default='demo_result', help='Output folder')
    parser.add_argument('-w', '--load_weights', type=str, default='./hybrid/weights/hybridnets.pth')
    parser.add_argument('--conf_thresh', type=restricted_float, default='0.25')
    parser.add_argument('--iou_thresh', type=restricted_float, default='0.3')
    parser.add_argument('--imshow',     type=boolean_string, default=False, help="Show result onscreen (unusable on colab, jupyter...)")
    parser.add_argument('--imwrite',    type=boolean_string, default=True, help="Write result to output folder")
    parser.add_argument('--show_det',   type=boolean_string, default=False, help="Output detection result exclusively")
    parser.add_argument('--show_seg',   type=boolean_string, default=False, help="Output segmentation result exclusively")
    parser.add_argument('--cuda',       type=boolean_string, default=True)
    parser.add_argument('--float16',    type=boolean_string, default=True, help="Use float16 for faster inference")
    parser.add_argument('--speed_test', type=boolean_string, default=False, help='Measure inference latency')
    args = parser.parse_args()

    params = Params(f'./hybrid/projects/{args.project}.yml')

    
    color_list_seg = {}
    for seg_class in params.seg_list:
        # edit your color here if you wanna fix to your liking
        color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))
    compound_coef = args.compound_coef
    source = args.source
    
    if source.endswith("/"):
        source = source[:-1]
    output = args.output
    if output.endswith("/"):
        output = output[:-1]

        
    weight = args.load_weights

    input_imgs = []
    shapes = []
    det_only_imgs = []

    anchors_ratios = params.anchors_ratios
    anchors_scales = params.anchors_scales

    threshold = args.conf_thresh
    iou_threshold = args.iou_thresh
    imshow = args.imshow
    imwrite = args.imwrite
    show_det = args.show_det
    show_seg = args.show_seg

    use_cuda = args.cuda
    use_float16 = args.float16
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = params.obj_list
    seg_list = params.seg_list

    color_list = standard_to_bgr(STANDARD_COLORS)
    

    img_name = f"sun_cast_{shadow_id}.png"
    img_path = os.path.join("NS_Images_Driver", img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        return

    ori_imgs = [frame]  # ✅ FIX: Initialize ori_imgs before using it

    ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]

    resized_shape = params.model['image_size']
    if isinstance(resized_shape, list):
        resized_shape = max(resized_shape)
    normalize = transforms.Normalize(
        mean=params.mean, std=params.std
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    for ori_img in ori_imgs:
        h0, w0 = ori_img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
        h, w = input_img.shape[:2]

        (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True,
                                                  scaleup=False)

        input_imgs.append(input_img)
        # cv2.imwrite('input.jpg', input_img * 255)
        shapes.append(((h0, w0), ((h / h0, w / w0), pad)))  # for COCO mAP rescaling

    if use_cuda:
        x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
    else:
        x = torch.stack([transform(fi) for fi in input_imgs], 0)

    x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)

    weight = torch.load(weight, map_location='cuda' if use_cuda else 'cpu')

    weight_last_layer_seg = weight['segmentation_head.0.weight']
    if weight_last_layer_seg.size(0) == 1:
        seg_mode = BINARY_MODE
    else:
        if params.seg_multilabel:
            seg_mode = MULTILABEL_MODE
        else:
            seg_mode = MULTICLASS_MODE
    #print("DETECTED SEGMENTATION MODE FROM WEIGHT AND PROJECT FILE:", seg_mode)
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
        # (B, W, H) -> (W, H)

        category = None
        for i in range(seg.size(0)):

            seg_mask = seg_mask_list[0]  # Only one mask in multiclass
            seg_mask_ = seg_mask[i].cpu().numpy()
            
            pad_h = int(shapes[i][1][1][1])
            pad_w = int(shapes[i][1][1][0])
            seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0] - pad_h, pad_w:seg_mask_.shape[1] - pad_w]
            seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)

            # Draw overlay, skipping road
            color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
            for index, seg_class in enumerate(params.seg_list):
                if seg_class.lower() == 'road':
                    continue
                color_seg[seg_mask_ == index + 1] = color_list_seg[seg_class]

            color_seg = color_seg[..., ::-1]  # RGB -> BGR
            color_mask = np.mean(color_seg, 2)
            det_only_imgs.append(ori_imgs[i].copy())
            seg_img = ori_imgs[i]
            seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
            seg_img = seg_img.astype(np.uint8)


            # ✅ Get all lane pixel coordinates (no save)
            lane_class_index = params.seg_list.index('lane') + 1
            lane_pixel_mask = (seg_mask_ == lane_class_index).astype(np.uint8)
            lane_pixel_coords = np.column_stack(np.where(lane_pixel_mask == 1))

            ########################################################
            lane_px = set(map(tuple, lane_pixel_coords))
            bright_px = set((y, x) for (x, y) in bright_pixels_driver)  # ✅ correct flip

            inter   = lane_px & bright_px
            overlap = len(inter) / len(bright_px) if bright_px else 0

            if (category == "Detected"):
                print("overlap: ", overlap)
            category = "Detected" if overlap > 0.1 else "Undetected"
            #print(f"[HybridNets] shadow_id={shadow_id} | bright:{len(bright_px)}  lane:{len(lane_px)}  → inter:{len(inter)} ({overlap:.2%}) → {category}")



            # print("len(bright_px):", len(bright_px))
            # print("len(lane_px):", len(lane_px))
            # print("len(inter):", len(inter))
            # print("Sample bright_px:", list(bright_px)[:5])
            # print("Sample lane_px:", list(lane_px)[:5])
            # print("Sample inter:", list(inter)[:5])
            # exit(0)



        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x,anchors, regression, classification,regressBoxes, clipBoxes,threshold, iou_threshold)

        for i in range(len(ori_imgs)):
            out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])
            for j in range(len(out[i]['rois'])):
                x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
                obj = obj_list[out[i]['class_ids'][j]]
                score = float(out[i]['scores'][j])

        # 1. Create output dirs
        os.makedirs("results/HybridNets/Detected", exist_ok=True)
        os.makedirs("results/HybridNets/Undetected", exist_ok=True)

        # 2. Save image based on category
        if imwrite:
            save_path = os.path.join("results", "HybridNets", category, f"sun_cast_{shadow_id}.png")
            cv2.imwrite(save_path, cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))


        return category == "Detected", overlap




# pixels= [(0,0),(1,1)]
# category = run_hybridnets(pixels, shadow_id=0)
# print(category)
#run_hybridnets("/home/tigersec/Projects/negative_shadow/Driver.png")