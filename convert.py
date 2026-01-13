import cv2
import numpy as np
from rknn.api import RKNN

# =========================
# 配置
# =========================
RKNN_MODEL = './model/yolov5m-seg-int.rknn'
ONNX_MODEL = './yolov5m-seg.onnx'
DATASET = './dataset.txt'
IMA_PATH = './street.jpg'
QUANTIZE_ON = True

OBJ_THRESH = 0.45
NMS_THRESH = 0.45
MAX_DETECT = 100
IMG_SIZE = (640, 640)

CLASSES = (
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
    "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
)

# CLASSES = ("0")

anchors = [
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]]
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def letter_box(img, new_shape):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(w * r), int(h * r)

    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape[1] - nw, new_shape[0] - nh
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return img, r, (left, top)

def class_aware_nms(boxes, scores, classes, iou_threshold=0.45, max_detect=300):
    if len(boxes) == 0:
        return []

    # 获取所有唯一类别
    unique_classes = np.unique(classes)
    keep = []

    for cls in unique_classes:
        mask = classes == cls
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]

        if len(cls_boxes) == 0:
            continue

        # 当前类别按分数排序
        order = cls_scores.argsort()[::-1]
        cls_keep = []

        while order.size > 0:
            i = order[0]
            cls_keep.append(i)

            if order.size == 1:
                break

            # IoU 计算
            xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
            yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
            xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
            yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
            area_o = (cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0]) * (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1])
            iou = inter / (area_i + area_o - inter + 1e-6)

            order = order[1:][iou <= iou_threshold]

        # 映射回全局索引
        global_indices = np.where(mask)[0][cls_keep]
        keep.extend(global_indices)

    # 关键修复：全局按分数重新排序，取 top max_detect
    if len(keep) > max_detect:
        final_scores = scores[keep]
        top_indices = final_scores.argsort()[::-1][:max_detect]
        keep = [keep[idx] for idx in top_indices]

    return keep

def crop_mask_numpy(masks, boxes, proto_size=160):
    print("[DEBUG] Entering crop_mask_numpy")
    N, H, W = masks.shape
    out_h, out_w = IMG_SIZE

    scale = proto_size / out_h
    boxes_proto = boxes * scale

    x1 = boxes_proto[:, 0][:, np.newaxis, np.newaxis]
    y1 = boxes_proto[:, 1][:, np.newaxis, np.newaxis]
    x2 = boxes_proto[:, 2][:, np.newaxis, np.newaxis]
    y2 = boxes_proto[:, 3][:, np.newaxis, np.newaxis]

    r = np.arange(W)[np.newaxis, np.newaxis, :]
    c = np.arange(H)[np.newaxis, :, np.newaxis]

    mask_keep = (r >= x1) & (r < x2) & (c >= y1) & (c < y2)

    masks_cropped = masks * mask_keep.astype(masks.dtype)

    masks_resized = np.zeros((N, out_h, out_w), dtype=np.float32)
    print(f"[DEBUG] Resizing {N} masks one by one...")
    for i in range(N):
        masks_resized[i] = cv2.resize(masks_cropped[i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    print("[DEBUG] crop_mask_numpy finished")
    return masks_resized > 0.5

def decode_boxes(raw_det, anchors_i, grid_h, grid_w, num_classes):
    """
    raw_det: (C, H, W) from RKNN
    anchors_i: anchor list for this scale
    """
    print(f"[DEBUG] decode_boxes started - grid: {grid_h}x{grid_w}")

    C, H, W = raw_det.shape
    assert H == grid_h and W == grid_w

    per_anchor = 5 + num_classes
    assert C % per_anchor == 0, \
        f"Channel {C} not divisible by (5+num_classes={per_anchor})"

    num_anchors = C // per_anchor
    print(f"[DEBUG] num_anchors={num_anchors}, num_classes={num_classes}")

    # reshape: (A, 5+cls, H, W)
    det = raw_det.reshape(num_anchors, per_anchor, grid_h, grid_w)

    gy, gx = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)

    stride_w = IMG_SIZE[1] / grid_w
    stride_h = IMG_SIZE[0] / grid_h

    boxes_list = []
    obj_list = []
    cls_list = []

    for a in range(num_anchors):
        aw, ah = anchors_i[a]
        pred = det[a]

        xy = pred[0:2] * 2.0 - 0.5 + np.stack((gx, gy), axis=0)
        xy *= np.array([stride_w, stride_h])[:, None, None]

        wh = (pred[2:4] * 2.0) ** 2
        wh *= np.array([aw, ah])[:, None, None]

        box = np.concatenate((xy - wh / 2, xy + wh / 2), axis=0)
        box = box.transpose(1, 2, 0).reshape(-1, 4)

        obj = sigmoid(pred[4]).reshape(-1)
        cls = sigmoid(pred[5:5+num_classes]) \
                .transpose(1, 2, 0).reshape(-1, num_classes)

        boxes_list.append(box)
        obj_list.append(obj)
        cls_list.append(cls)

    boxes = np.concatenate(boxes_list, axis=0)
    obj_conf = np.concatenate(obj_list, axis=0)
    cls_conf = np.concatenate(cls_list, axis=0)

    print(f"[DEBUG] decode_boxes finished - total boxes: {boxes.shape[0]}")
    return boxes, obj_conf, cls_conf


def post_process(outputs, anchors):
    print("[DEBUG] === post_process START ===")
    print(f"[DEBUG] Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  output[{i}].shape = {out.shape}")

    proto = outputs[6][0]
    print(f"[DEBUG] proto shape: {proto.shape}")

    detect_outputs = [outputs[0][0], outputs[2][0], outputs[4][0]]
    seg_outputs    = [outputs[1][0], outputs[3][0], outputs[5][0]]

    sizes = [(80,80), (40,40), (20,20)]

    all_boxes = []
    all_obj_conf = []
    all_cls_conf = []
    all_seg_coeffs = []

    for i in range(3):
        print(f"\n[DEBUG] >>> Processing feature layer {i} (size {sizes[i]}) <<<")
        det = detect_outputs[i]
        seg = seg_outputs[i]
        grid_h, grid_w = sizes[i]

        # seg coeff
        print("[DEBUG] Computing seg_coeff...")
        seg_coeff = seg.reshape(3, 32, grid_h, grid_w)
        seg_coeff = seg_coeff.transpose(0, 2, 3, 1).reshape(-1, 32)
        print(f"[DEBUG] seg_coeff shape: {seg_coeff.shape}")

        # decode boxes
        num_classes = len(CLASSES)
        boxes, obj_conf, cls_conf = decode_boxes(det, anchors[i], grid_h, grid_w, num_classes)

        all_boxes.append(boxes)
        all_obj_conf.append(obj_conf)
        all_cls_conf.append(cls_conf)
        all_seg_coeffs.append(seg_coeff)

    print("\n[DEBUG] Concatenating all layers...")
    boxes = np.concatenate(all_boxes, axis=0)
    obj_conf = np.concatenate(all_obj_conf, axis=0)
    cls_conf = np.concatenate(all_cls_conf, axis=0)
    seg_coeffs = np.concatenate(all_seg_coeffs, axis=0)
    print(f"[DEBUG] Total raw detections: {boxes.shape[0]}")

    print("[DEBUG] Applying confidence threshold...")
    class_scores = cls_conf.max(axis=1) * obj_conf
    keep = class_scores >= OBJ_THRESH
    print(f"[DEBUG] Detections after threshold: {keep.sum()}")

    if not np.any(keep):
        print("[DEBUG] No detections after threshold!")
        return None, None, None, None

    boxes = boxes[keep]
    scores = class_scores[keep]
    classes = cls_conf.argmax(axis=1)[keep]
    seg_coeffs = seg_coeffs[keep]

    print(f"[DEBUG] Before NMS: {len(boxes)} detections")
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=NMS_THRESH, max_detect=MAX_DETECT)
    print(f"[DEBUG] After NMS: {len(keep)} detections")

    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    seg_coeffs = seg_coeffs[keep]

    print("[DEBUG] Generating instance masks...")
    proto_flat = proto.reshape(32, -1)
    masks = seg_coeffs @ proto_flat
    masks = sigmoid(masks)
    masks = masks.reshape(-1, 160, 160)
    print(f"[DEBUG] Generated {masks.shape[0]} raw masks (160x160)")

    print("[DEBUG] Cropping and resizing masks to 640x640...")
    masks = crop_mask_numpy(masks, boxes)
    print("[DEBUG] Masks processing completed")

    print("[DEBUG] === post_process FINISHED ===")
    return boxes, classes, scores, masks.astype(bool)

def restore_boxes(boxes, src_shape, ratio, pad):
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= ratio
    h, w = src_shape
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
    return boxes

def restore_masks(masks, src_shape, pad):
    """
    masks: (N, 640, 640) bool
    pad: (left, top) from letter_box
    src_shape: (orig_h, orig_w)
    """
    N, H, W = masks.shape  # 640, 640
    orig_h, orig_w = src_shape
    left, top = pad

    # 计算有效区域（去除 letterbox 的黑边）
    effective_h = H - 2 * top   # 通常 top == bottom
    effective_w = W - 2 * left  # 通常 left == right

    # 裁剪掉 padding
    cropped = masks[:, top : top + effective_h, left : left + effective_w]  # (N, effective_h, effective_w)

    # 如果原图尺寸和有效区域一致，直接转 uint8 返回
    if effective_h == orig_h and effective_w == orig_w:
        return cropped.astype(np.uint8)

    # 需要 resize：逐个处理，但先转 uint8
    restored = np.zeros((N, orig_h, orig_w), dtype=np.uint8)
    print(f"[DEBUG] Restoring {N} masks to original size {orig_h}x{orig_w}...")
    for i in range(N):
        # bool -> uint8 后再 resize（解决 cv2 报错）
        temp = cropped[i].astype(np.uint8) * 255  # 可选：乘 255 让 mask 更明显（白底黑掩码变白掩码）
        restored[i] = cv2.resize(temp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    print("[DEBUG] restore_masks finished")
    return restored

def merge_seg(image, masks, classes, alpha=0.5):
    print("[DEBUG] Starting merge_seg (optimized version)...")
    out = image.copy().astype(np.float32)  # 用 float32 避免溢出

    colors = np.array([(0,0,255),(255,128,0),(255,255,0),(0,255,0),
                       (0,255,255),(255,0,0),(128,0,255),(255,0,255),
                       (255,165,0),(128,128,128)] * 10, dtype=np.float32)

    for i in range(len(masks)):
        color = colors[int(classes[i]) % len(colors)]
        mask = masks[i].astype(bool)
        out[mask] = out[mask] * (1 - alpha) + color * alpha

    print("[DEBUG] merge_seg finished")
    return np.clip(out, 0, 255).astype(np.uint8)

def draw(img, boxes, scores, classes):
    for b, s, c in zip(boxes, scores, classes):
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(img, f'{CLASSES[c]} {s:.2f}', (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

def export_rknn_inference(img):
    rknn = RKNN(verbose=True)
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], quantized_algorithm='normal', quantized_method='channel', target_platform='rk3588')
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')

    return outputs

if __name__ == '__main__':
    print("[DEBUG] Program started")
    img_src = cv2.imread(IMA_PATH)
    if img_src is None:
        print("Error: Cannot load image")
        exit()
    print(f"[DEBUG] Image loaded, shape: {img_src.shape}")

    img_lb, ratio, pad = letter_box(img_src, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    print(f"[DEBUG] Letterbox done, ratio={ratio}, pad={pad}")

    outputs = export_rknn_inference(img_rgb)

    print("[DEBUG] Starting post_process...")
    boxes, classes, scores, masks = post_process(outputs, anchors)

    if boxes is not None and len(boxes) > 0:
        print(f"[DEBUG] Final detections: {len(boxes)}")
        boxes = restore_boxes(boxes, img_src.shape[:2], ratio, pad)
        masks = restore_masks(masks, img_src.shape[:2], pad)

        img_out = merge_seg(img_src, masks, classes, alpha=0.5)
        draw(img_out, boxes, scores, classes)
        cv2.imwrite('result.jpg', img_out)
        print("Saved result.jpg with instance segmentation!")
    else:
        print("No objects detected.")
    print("[DEBUG] Program finished")
