import os
import shutil
import cv2 as cv
from PIL import Image
from utils.util import image_compose_with_margin, create_none_matrix, image_compose


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


def main():
    # test1()
    # test2()
    test3()


if __name__ == '__main__':
    main()
