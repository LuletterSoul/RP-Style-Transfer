import cv2
import os
import numpy as np

content_dir = '/data/lxd/datasets/photo_data/content'
style_dir = '/data/lxd/datasets/photo_data/style'

content_names = os.listdir(content_dir)
style_names = os.listdir(style_dir)


exp_name = '1127_SpadeRPNet_lr1e-3_cw1_sw1'
test_num = '20000'

input = f'output/{exp_name}/test/{test_num}'
output = f'output/{exp_name}/test/{test_num}'

input_names = os.listdir(input)

images = []
for idx, name in enumerate(input_names):
    img = cv2.imread(os.path.join(input, name))
    images.append(img)
    print(f'Processing {name}.')
    if (idx + 1) % 30 == 0:
        images = np.vstack(images)
        cv2.imwrite(os.path.join(output, f'{idx + 1}.png'), images)
        images = []
