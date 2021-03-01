import torch
from PIL import Image


def make_grid(reference_img, imgs, w_num, span=0, unit_size=512):
    h_num = len(imgs) // w_num
    w = w_num * (unit_size + span)
    h = h_num * (unit_size + span)
    whole_shape = ((w_num + 1) * (unit_size + span), h)  # 全部区域，比主体区域多一列
    target_shape = (w, h)  # 主体区域
    ref_shape = (unit_size + span, h)  # 参考图区域；第一列，从上往下第一张图是参考图，其余为空白
    target = Image.new('RGB', target_shape, (255, 255, 255))
    whole = Image.new('RGB', whole_shape, (255, 255, 255))
    reference = Image.new('RGB', ref_shape, (255, 255, 255))
    width = 0
    for img in imgs:
        x, y = int(width % target_shape[0]), int(
            width/target_shape[0])*(unit_size+span)  # 左上角坐标，从左到右递增
        target.paste(img.resize(
            (unit_size, unit_size)), (x, y, x+unit_size, y+unit_size))
        width += (unit_size+span)
    reference.paste(reference_img.resize((unit_size, unit_size)),
                    (0, 0, unit_size, unit_size))
    whole.paste(reference, (0, 0, ref_shape[0], ref_shape[1]))
    whole.paste(target, (unit_size + span, 0,
                         whole_shape[0], whole_shape[1]))
    return whole
