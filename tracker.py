from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from PIL import Image
import random
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

id_set= {}
P=0.02

def plot_bboxes(image, bboxes, attribute_model, decoder, use_id, transforms,indice_system, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        src = Image.fromarray(image[max(y1 - 2,0):min(y2 + 2,image.shape[0]), max(x1 - 2,0):min(x2 + 2,image.shape[1]), :])
        src = transforms(src)
        src = src.unsqueeze(dim=0)
        if pos_id in id_set.keys():
            seed=random.random()
            if seed>1-P:
                if not use_id:
                    out = attribute_model.forward(src)
                else:
                    out, _ = attribute_model.forward(src)
                pred = torch.gt(out, torch.ones_like(out) / 2)  # threshold=0.5
                res = decoder.decode(pred)
                f = True
                for k in indice_system.keys():
                    if k in res.keys():
                        if res[k] != indice_system[k]:
                            f = False
                # 红色为目标值
                if f:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                id_set[pos_id] = color
            else:
                pass
        else:
            if not use_id:
                out = attribute_model.forward(src)
            else:
                out, _ = attribute_model.forward(src)
            pred = torch.gt(out, torch.ones_like(out) / 2)  # threshold=0.5
            res=decoder.decode(pred)
            f=True
            for k in indice_system.keys():
                if k in res.keys():
                    if res[k]!=indice_system[k]:
                        f=False
            #红色为目标值
            if f:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            id_set[pos_id]=color

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, id_set[pos_id], thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, id_set[pos_id], -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def update_tracker(target_detector, image):
    new_faces = []
    _, bboxes = target_detector.detect(image)


    bbox_xywh = []
    confs = []
    clss = []
    for x1, y1, x2, y2, cls_id, conf in bboxes:
        obj = [
            int((x1 + x2) / 2), int((y1 + y2) / 2),
            x2 - x1, y2 - y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, clss, image)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
        current_ids.append(track_id)
        if cls_ == 'face':
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2]
                new_faces.append((face, track_id))
            face_bboxes.append(
                (x1, y1, x2, y2)
            )

    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        if target_detector.faceTracker[history_id] < -5:
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
        print('-[INFO] Delete track id:', ids)

    image = plot_bboxes(image, bboxes2draw, target_detector.a, target_detector.d, target_detector.flag,
                        target_detector.t,target_detector.indice_system)

    return image, new_faces, face_bboxes
