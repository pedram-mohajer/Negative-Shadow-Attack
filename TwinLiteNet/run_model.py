import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import os
import torch
from model import TwinLite as net
import cv2

def Run(model,img):
    img = cv2.resize(img, (640, 360))
    img_rs=img.copy()
    img_seg=img.copy()
    img_lane=img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    x0=img_out[0]
    x1=img_out[1]

    _,da_predict=torch.max(x0, 1)
    _,ll_predict=torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0]*255
    LL = ll_predict.byte().cpu().data.numpy()[0]*255
    img_seg[DA>100]=[255,0,0]
    img_lane[LL>100]=[0,255,0]
    # img_seg = cv2.resize(img_seg, (1640, 590))
    img_rs = img_lane
    # img_rs[DA>100]=[255,0,0]
    # img_rs[LL>100]=[0,255,0]
    
    return img_rs, img_seg, img_lane


if __name__ == "__main__":
    print('Loading Model')                  
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('pretrained/best.pth'))
    model.eval()

    for subdir, _, files in os.walk('data/input'):
        if files:
            parent_dir = os.path.basename(os.path.dirname(subdir))
            current_dir = os.path.basename(subdir)
            if parent_dir != 'input':
                print(f'Proccessing twinlitenet files in {parent_dir}/{current_dir}')
                output_path = os.path.join('data/output', f'{parent_dir}/{current_dir}')
            else:
                print(f'Processing twinlitenet files in {current_dir}')
                output_path = os.path.join('data/output', current_dir)
            
            os.makedirs(output_path, exist_ok=True)

            for file in files:
                img = cv2.imread(os.path.join(subdir, file))
                img_rs, _, _ = Run(model, img)
                cv2.imwrite(os.path.join(output_path, file), img_rs)