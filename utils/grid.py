import os
from pathlib import Path
import shutil
import cv2 
from PIL import Image
import traceback
import numpy as np

from PIL import Image
import re

def tryint(s):  # 将元素中的数字转换为int后再排序
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):  # 将元素中的字符串和数字分割开
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def sort_humanly(v_list):  # 以分割后的list为单位进行排序
    """
    sort list strings according string and number
    :param v_list:
    :return:
    """
    return sorted(v_list, key=str2int, reverse=False)

def image_compose(img_path_matrix, img_saved_path, img_size=512):
    row, col = len(img_path_matrix), len(img_path_matrix[0])
    res = Image.new('RGB', (col * img_size, row * img_size))  # (width, height)
    for i in range(row):
        for j in range(col):
            img_path = img_path_matrix[i][j]
            print(f'process {img_path} ...')
            if img_path is not None:
                try:
                    temp_img = Image.open(img_path).resize((img_size, img_size), Image.ANTIALIAS)
                    res.paste(temp_img, box=(img_size * j, img_size * i))  # -x |y
                except Exception as e:
                    print(e)
    res.save(img_saved_path)


def image_compose_with_margin(img_path_matrix, img_saved_path, img_size, margin):
    row, col = len(img_path_matrix), len(img_path_matrix[0])  # 行， 列
    h_sum, w_sum = img_size[1] * row + (row - 1) * margin, img_size[0] * col + (col - 1) * margin
    res_img = Image.new('RGB', (w_sum, h_sum), (255, 255, 255))
    for i in range(row):
        for j in range(col):
            img_path = img_path_matrix[i][j]
            print(f'process {img_path}...')
            if img_path is not None:
                try:
                    temp_img = Image.open(img_path).convert('RGB').resize((img_size[0], img_size[1]), Image.ANTIALIAS)
                    box = (j * (img_size[0] + margin), i * (img_size[1] + margin))
                    res_img.paste(temp_img, box=box)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
    res_img.save(img_saved_path)
    print(f'[{img_saved_path}] saved ...')


def create_none_matrix(row, col):
    img_path_matrix = [[None for _ in range(col)] for _ in range(row)]
    return img_path_matrix


def print_matrix(img_path_matrix):
    for path_list in img_path_matrix:
        print(path_list)


def move_and_resize_img(src_path, dst_path):
    img = Image.open(src_path)
    img_resized = img.resize((512, 512))
    img_resized.save(dst_path)

def test1():
    res_dir = [
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/test/style-projection/2020-11-30-4/r41-cycle4-patch3',
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/test/style-projection/2020-11-30-4/r41-cycle5-patch3',
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/test/style-projection/2020-11-30-4/r41-cycle6-patch3',
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/test/style-projection/2020-11-30-4/r41-no-cycle-patch3']
    img_name_list = os.listdir(res_dir[0])
    img_saved_dir = '../result/2020-11-30-1'
    os.makedirs(img_saved_dir, exist_ok=True)

    for img_name in img_name_list:
        img_path_matrix = create_none_matrix(row=1, col=len(res_dir))
        for j in range(len(res_dir)):
            img_path_matrix[0][j] = os.path.join(res_dir[j], img_name)
        img_saved_path = os.path.join(img_saved_dir, img_name)
        image_compose_with_margin(img_path_matrix, img_saved_path, img_size=[512, 3072], margin=4)


def test2():
    res_dir = [
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/train/cycle7/2020-12-2-1/test',
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/train/cycle7/2020-12-2-2/test']
    img_name_list = os.listdir(res_dir[0])
    img_saved_dir = '../result/2020-12-3-1'
    os.makedirs(img_saved_dir, exist_ok=True)

    for img_name in img_name_list:
        img_path_matrix = create_none_matrix(row=1, col=len(res_dir))
        for j in range(len(res_dir)):
            img_path_matrix[0][j] = os.path.join(res_dir[j], img_name)
        img_saved_path = os.path.join(img_saved_dir, img_name)
        image_compose_with_margin(img_path_matrix, img_saved_path, img_size=[512, 3072], margin=4)


def test3():
    res_dir = [
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/test/style-projection/2020-11-29-1/r41-no-cycle-patch3',
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/train/cycle13/2020-12-30-1/test',
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/train/cycle13/20201-1-4-2/test'
    ]
    img_name_list = os.listdir(res_dir[0])
    img_saved_dir = '../result/2021-1-7-1'
    os.makedirs(img_saved_dir, exist_ok=True)

    for img_name in img_name_list:
        img_path_matrix = create_none_matrix(row=1, col=len(res_dir))
        for j in range(len(res_dir)):
            img_path_matrix[0][j] = os.path.join(res_dir[j], img_name)
        img_saved_path = os.path.join(img_saved_dir, img_name)
        image_compose_with_margin(img_path_matrix, img_saved_path, img_size=[512, 3072], margin=4)

def test1():
    res_dir = [
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/test/style-projection/2020-11-30-4/r41-cycle4-patch3',
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/test/style-projection/2020-11-30-4/r41-cycle5-patch3',
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/test/style-projection/2020-11-30-4/r41-cycle6-patch3',
        'D:/Projects/Python/StyleTransfer/StyleProjection/results/test/style-projection/2020-11-30-4/r41-no-cycle-patch3']
    img_name_list = os.listdir(res_dir[0])
    img_saved_dir = '../result/2020-11-30-1'
    os.makedirs(img_saved_dir, exist_ok=True)

    for img_name in img_name_list:
        img_path_matrix = create_none_matrix(row=1, col=len(res_dir))
        for j in range(len(res_dir)):
            img_path_matrix[0][j] = os.path.join(res_dir[j], img_name)
        img_saved_path = os.path.join(img_saved_dir, img_name)
        image_compose_with_margin(img_path_matrix, img_saved_path, img_size=[512, 3072], margin=4)

def is_valid_img_path(path):
    basename = os.path.basename(path)
    print(path)
    return os.path.isfile(path) and 'cat' not in basename

def compose_compared_imgs(content_dir,style_dir, compared_method_dirs,output_dir,selected_pairs=None):
    content_paths = [os.path.join(content_dir,c) for c in sort_humanly(os.listdir(content_dir))]
    style_paths = [os.path.join(style_dir, s) for s in sort_humanly(os.listdir(style_dir))]
    compared_method_paths = [sort_humanly([str(p) for p in list(Path(method_name).glob('*')) if is_valid_img_path(str(p))]) for method_name in compared_method_dirs]
    pair_paths = list(zip(content_paths, style_paths, *compared_method_paths))
    os.makedirs(output_dir,exist_ok=True)
    whole_img = []
    cnt = 0
    for idx, pairs in enumerate(pair_paths):
        pairs = list(pairs)
        selected = False
        if selected_pairs:
            for p in pairs:
                if os.path.basename(p) in selected_pairs:
                    selected = True
        else:
            selected = True

        if not selected: 
            continue
        row_img = np.hstack([cv2.resize(cv2.imread(p),(512,512)) for p in pairs])
        whole_img = np.vstack([whole_img, row_img]) if len(whole_img) else np.vstack([row_img])
        print(f'Processing {idx}: {pairs}')
        if (cnt+1) % 10 == 0:
            cv2.imwrite(os.path.join(output_dir, f'{idx+1}.png'), whole_img)
            whole_img = []
        cnt +=1
    if len(whole_img):
       cv2.imwrite(os.path.join(output_dir, f'last.png'), whole_img)

def crop_original(method_dirs):
    for method in method_dirs:
        names = os.listdir(method)
        for n in names:
            prefix = os.path.splitext(n)[0]
            img_path = os.path.join(method, n)
            if not os.path.isfile(img_path):
                continue
            img = cv2.imread(img_path)
            img = img[:, 512 * 2:,:]
            new_dir = os.path.join(method,'crop')
            os.makedirs(new_dir, exist_ok=True)
            cv2.imwrite(os.path.join(new_dir,n), img)
    
def compared_rp_net():
    content_dir = '/data/lxd/datasets/photo_data/content'
    style_dir = '/data/lxd/datasets/photo_data/style'
    # compared_method_dirs = ['output/1126_RPNet_ST_lr1e-3_cw1_sw2_mw0/test/120000/crop', 
    #                         'output/0106_AdaINRPNet_lr1e-3_cw1_sw1/test/110000/crop',
    #                         'output/0108_MultiAdaINRPNet_rp8_3incep_lr1e-4_cw1_sw1/test/340000/crop',
    #                         'output/0109_LDAdaINRPNet_lr1e-3_cw1_sw1/test/200000/crop',
    #                         'output/baselines/DPST', 
    #                         'output/baselines/LST', 
    #                         'output/baselines/PhotoWCT',
    #                         'output/baselines/WCT2']
    # compared_method_dirs = ['output/1126_RPNet_ST_lr1e-3_cw1_sw2_mw0/test/120000/crop', 
    #                         'output/0106_AdaINRPNet_lr1e-3_cw1_sw1/test/110000/crop',
    #                         'output/0108_MultiAdaINRPNet_rp8_3incep_lr1e-4_cw1_sw1/test/340000/crop',
    #                         'output/0109_LDAdaINRPNet_lr1e-3_cw1_sw1/test/200000/crop',
    #                         'output/0115_MultiAdaINRPNet_rp3_incep0_lr1e-4_hiddim16_cw1_sw1/test/150000',
    #                         'output/0115_MultiAdaINRPNet_rp3_incep0_lr1e-3_hiddim32_cw1_sw1/test/190000',
    #                         'output/0113_WCTRPNet_lr1e-4_cw1_sw1/test/140000',
    #                         'output/0115_MultiAdaINRPNet_rp8_3incep_lr1e-4_cw1_sw1_test/test/1',
    #                         'output/0116_MultiAdaINRPNet_constant_rp10_incep3_lr1e-4_hiddim32_cw1_sw1/test/35000'
    #                         ]                            
    compared_method_dirs = ['output/0115_MultiAdaINRPNet_rp8_3incep_lr1e-4_cw1_sw1_test/test/1',
                            'output/0116_MultiAdaINRPNet_constant_rp10_incep3_lr1e-4_hiddim32_cw1_sw1/test/35000',
                                                        'output/baselines/DPST', 
                            'output/baselines/LST', 
                            'output/baselines/PhotoWCT',
                            'output/baselines/WCT2'
                            ]     
    # compared_method_dirs = ['output/1126_RPNet_ST_lr1e-3_cw1_sw2_mw0/test/120000', 
    #                     'output/0106_AdaINRPNet_lr1e-3_cw1_sw1/test/110000',
    #                     'output/0108_MultiAdaINRPNet_rp8_3incep_lr1e-4_cw1_sw1/test/340000',
    #                     'output/0109_LDAdaINRPNet_lr1e-3_cw1_sw1/test/200000',
    #                     'output/baselines/DPST', 
    #                     'output/baselines/LST', 
    #                     'output/baselines/PhotoWCT',
    #                     'output/baselines/WCT2']
    output_dir = 'output/compared/0116_rpnet_mask_cmp'
    # crop_original(compared_method_dirs[0:4])
    compose_compared_imgs(content_dir,style_dir,compared_method_dirs, output_dir)

def compared_adaptive_sanet():
    content_dir = '/data/lxd/datasets/default_st/test1/content'
    style_dir = '/data/lxd/datasets/default_st/test1/style'
    compared_method_dirs = ['output/0107_AdaptiveSANet_lr1e-3_cw1_sw1/test/220000/crop', 
                            'output/0107_StaticSANet_lr1e-4_cw1_sw3_identw50_identw1/test/420000/crop',
                            'output/0114_AdaptiveSANet_lr1e-4_cw1_sw1_fval0.4_visulized/test/180000',
                            'output/0115_AdaptiveSANet_lr1e-4_cw1_sw1_relu/test/245000'
                            ]
    output_dir = 'output/compared/0116_adaptive_selected_sanet_cmp'
    # crop_original(compared_method_dirs[0:2])
    compose_compared_imgs(content_dir,style_dir,compared_method_dirs, output_dir,selected_pairs=['0-0.png','3-3.png','8-8.png','9-9.png','20-20.png'])
    # compose_compared_imgs(content_dir,style_dir,compared_method_dirs, output_dir)





def main():
    # compared_rp_net()
    compared_adaptive_sanet()
 


if __name__ == '__main__':
    main()
    # base_dir = 'output/baselines/PhotoWCT'
    # names = os.listdir(base_dir)
    # for n in names:
    #     path = os.path.join(base_dir,n)
    #     if os.path.exists(path) and os.path.isfile(path):
    #         img = cv2.imread(path)
    #         prefix = os.path.splitext(n)[0]
    #         cv2.imwrite(os.path.join(base_dir,f'{prefix}.'))
            
