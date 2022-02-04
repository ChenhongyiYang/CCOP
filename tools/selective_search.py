import json
import os
import cv2
import numpy as np

def remove_aspect_ratio(rects, thr1=1./3, thr2=3.):
    r = rects[:, 2] / rects[:, 3]
    return rects[(r > thr1) & (r < thr2)]

def remove_small(rects, thr=12.):
    size = np.sqrt(rects[:, 2] * rects[:, 3])
    return rects[size > thr]

def get_iou(tbox, rest_box):
    tbox = np.reshape(tbox, [1, 5])
    rvol = (rest_box[:, 2] - rest_box[:, 0]) * (rest_box[:, 3] - rest_box[:, 1])
    tvol = (tbox[:, 2] - tbox[:, 0]) * (tbox[:, 3] - tbox[:, 1])

    iymin = np.maximum(tbox[:, 0], rest_box[:, 0])
    ixmin = np.maximum(tbox[:, 1], rest_box[:, 1])
    iymax = np.minimum(tbox[:, 2], rest_box[:, 2])
    ixmax = np.minimum(tbox[:, 3], rest_box[:, 3])

    ih = np.maximum(iymax - iymin, 0.)
    iw = np.maximum(ixmax - ixmin, 0.)
    ivol = ih * iw

    iou = ivol / (rvol + tvol - ivol)
    return iou

def nms_cpu(boxes, scores, Nt, threshold):
    if scores.shape[0] == 0:
        return None, None
    i = 0

    _boxes = np.copy(boxes)
    _scores = np.copy(scores)

    box_ind = np.arange(_boxes.shape[0]).reshape([-1, 1])
    _boxes = np.concatenate((_boxes, box_ind), axis=1)
    while i < _boxes.shape[0] - 1:
        max_pos = np.argmax(_scores[i:]) + i

        tscore = _scores[max_pos]
        tbox = _boxes[max_pos, :].copy()
        _scores[max_pos] = _scores[i]
        _scores[i] = tscore

        _boxes[max_pos, :] = _boxes[i, :]
        _boxes[i, :] = tbox

        rest_box = _boxes[i+1:, :]
        tbox = np.reshape(tbox, [1,5])

        iou = get_iou(tbox, rest_box)
        # merged_boxes = np.concatenate((tbox.reshape(1, 5), rest_box[iou>Nt]), axis=0).mean(axis=0)
        # _boxes[i, :4] = merged_boxes[:4]

        merged_boxes = np.concatenate((tbox.reshape(1, 5), rest_box[iou>Nt]), axis=0).mean(axis=0)
        _boxes[i, :4] = tbox[:4]

        w = np.less_equal(iou, Nt).astype(np.float32)
        _scores[i+1:] = _scores[i+1:] * w
        inds = np.where(_scores > threshold)
        _scores = _scores[inds]
        _boxes = _boxes[inds]
        i += 1

    got_inds = _boxes[:,4].reshape([-1]).astype(np.int32)
    ret_scores = scores[got_inds]
    ret_boxes = boxes[got_inds]
    return ret_boxes, ret_scores

def selective_search(img_paths, keepnum=50):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    all_boxes = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        ss.setBaseImage(img)
        # ss.switchToSelectiveSearchQuality()
        ss.switchToSelectiveSearchFast()
        rects = ss.process()

        rects = np.array(rects).astype(np.int32)
        rects = remove_aspect_ratio(rects)
        rects = remove_small(rects, thr=20)
        boxes = np.concatenate((rects[:, :2], rects[:, :2]+rects[:, 2:]), axis=1)  # xywh -> xyxy
        scores = np.random.uniform(0.01, 1., boxes.shape[0])
        boxes, _ = nms_cpu(boxes, scores, 0.5, 0.)
        if boxes is not None:
            boxes = np.concatenate((boxes[:, :2], boxes[:,2:]-boxes[:,:2]), axis=1)  # xyxy -> xywh
            all_boxes.append(boxes[:keepnum].tolist())
        else:
            all_boxes.append([])
    return all_boxes

def write_json(images, all_boxes, out_path):
    data = {}
    data['images'] = []
    for img in images:
        data['images'].append({
            'id': img['id'],
            'width': img['width'],
            'height':img['height'],
            'file_name':img['file_name']}
        )
    data['annotations'] = []
    data['categories'] = [{'supercategory': 'RoI', 'id': 1, 'name': 'RoI'}]

    box_id = 0
    for img, boxes in zip(images, all_boxes):
        for box in boxes:
            ann = {
                'image_id': img['id'],
                'bbox': box,
                'category_id': 1,
                'id': box_id
            }
            box_id += 1
            data['annotations'].append(ann)
    with open(out_path, 'w') as f:
        json.dump(data, f)

def merge_json(json_roots, out_path):
    data = {}
    data['images'] = []
    data['annotations'] = []
    data['categories'] = [{'supercategory': 'RoI', 'id': 1, 'name': 'RoI'}]

    ann_id = 0
    for json_file in os.listdir(json_roots):
        with open(os.path.join(json_roots, json_file)) as f:
            single_data = json.load(f)
        data['images'] = data['images'] + single_data['images']
        for ann in single_data['annotations']:
            ann['id'] = ann_id
            data['annotations'].append(ann)
            ann_id += 1

    with open(out_path, 'w') as f:
        json.dump(data, f)



if __name__ == '__main__':
    import multiprocessing as mp
    import time

    img_root = 'path_to_coco/train2017'
    json_file = 'path_to_coco/annotations/instances_train2017.json'
    out_root = 'path_to_coco/annotations/ss_train2018.json'

    with open(json_file) as f:
        data = json.load(f)
    images = data['images']

    start = 0
    end = len(images)

    images_start_end = images[start:end]
    all_paths = [x['file_name'] for x in images_start_end]

    pcount = max(mp.cpu_count(), 32)
    pool = mp.Pool(pcount)
    img_path_maps = [[] for _ in range(pcount)]
    for i, img_path in enumerate(all_paths):
        img_path_maps[i % pcount].append(os.path.join(img_root, img_path))
    all_boxes_maps = pool.map(selective_search, img_path_maps)

    all_boxes = []
    for boxes_maps in all_boxes_maps:
        all_boxes = all_boxes + boxes_maps

    write_json(images_start_end, all_boxes, out_root)

