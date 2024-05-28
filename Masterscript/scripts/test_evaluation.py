import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
## pink is lane that attack-control lost
## green is lane that attack-control added
input_img = cv2.imread('./Shadow_Attack/masterscript/data/input/test/back_edge.png')
preld_img = cv2.imread('./Shadow_Attack/masterscript/data/output/overlaid_imgs/test_overlaid/1/back_edge.png')
control_img = cv2.imread('./Shadow_Attack/masterscript/data/evaluation/hybridnets/test/back_edge.png')
attack_img = cv2.imread('./Shadow_Attack/masterscript/data/evaluation/hybridnets/test_overlaid/1/back_edge.png')

CROP_H = 540
CROP_W_L = 20
CROP_W_R = 1600

print((CROP_W_R-CROP_W_L) * CROP_H * 0.05)
CUTOFF = 8000
x,y,_ = attack_img.shape
print(f'{x},{y}')
preld_img = cv2.resize(preld_img, (y,x))
input_img = cv2.resize(input_img, (y,x))


preld_img = preld_img[:CROP_H, CROP_W_L:CROP_W_R]
control_img = control_img[:CROP_H, CROP_W_L:CROP_W_R]
attack_img = attack_img[:CROP_H, CROP_W_L:CROP_W_R]
input_img = input_img[:CROP_H, CROP_W_L:CROP_W_R]



# attack_img = cv2.medianBlur(attack_img,5)
diff_preld_attack = cv2.subtract(preld_img, attack_img)
diff_input_control = cv2.subtract(input_img, control_img)
diff_plda_incnt= cv2.subtract(diff_preld_attack, diff_input_control)
diff_incnt_plda = cv2.subtract(diff_input_control, diff_preld_attack)

# diff = cv2.absdiff(preld_img, diff)
# diff = cv2.subtract(cv2.subtract(preld_img,attack_img), control_img)
gdiff_plda_incnt = cv2.cvtColor(diff_plda_incnt ,cv2.COLOR_BGR2GRAY)
gdiff_inct_plda = cv2.cvtColor(diff_incnt_plda, cv2.COLOR_BGR2GRAY)
gdiff_pld_att= cv2.cvtColor(diff_preld_attack, cv2.COLOR_BGR2GRAY)
gdiff_in_cnt = cv2.cvtColor(diff_input_control, cv2.COLOR_BGR2GRAY)




# diff2 = cv2.subtract(diff_b, diff_a)
# diff2 = cv2.subtract(diff2, preld_img)


diff_t = cv2.adaptiveThreshold(gdiff_plda_incnt, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
diff2_t = cv2.adaptiveThreshold(gdiff_inct_plda, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)

# diff_range = cv2.inRange(diff_c, np.array([0, 90, 0]), np.array([100,255,100]))
print(f'tot size {gdiff_plda_incnt.shape[0] * gdiff_plda_incnt.shape[1]}')
print(f'Cutoff: {CUTOFF}')
print(f'count nz nonthresh= {cv2.countNonZero(diff2_t)}')
print(f'count nz thresh= {cv2.countNonZero(diff_t)}')
print(f'Success added: {cv2.countNonZero(diff_t)>CUTOFF}')
print(f'Success removed: {cv2.countNonZero(diff2_t)>CUTOFF}')

plt.figure()
plt.title('input')
plt.imshow(input_img[...,::-1])
plt.draw()

plt.figure()
plt.title('preld')
plt.imshow(preld_img[...,::-1])
plt.draw()

plt.figure()
plt.title('control')
plt.imshow(control_img[...,::-1])
plt.draw()

plt.figure()
plt.title('attack')
plt.imshow(attack_img[...,::-1])
plt.draw()

plt.figure()
plt.title('diff_prelda-attack')
plt.imshow(gdiff_pld_att, 'gray')
plt.draw()

plt.figure()
plt.title('diff_input_control')
plt.imshow(gdiff_in_cnt, 'gray')
plt.draw()

plt.figure()
plt.title('diffplda incnt')
plt.imshow(gdiff_plda_incnt, 'gray')
plt.draw()

plt.figure()
plt.title('diffincnt plda')
plt.imshow(diff_incnt_plda, 'gray')
plt.draw()

plt.figure()
plt.title('plda int threshold')
plt.imshow(diff_t, 'gray')
plt.draw()
plt.figure()
plt.title('int plda threshold')
plt.imshow(diff2_t, 'gray')
plt.draw()


plt.show()