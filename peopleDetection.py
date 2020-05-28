import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

import time

device = torch.device('cpu')
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'painting', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def inizializeModel(id):
    if id == 1:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif id == 2:
        model = get_instance_segmentation_model(2)
        model.load_state_dict(torch.load('parameters.pt'))
    else:
        return None
    model.eval()
    model.to(device)
    return model


def peopleDetection(frame, model, threshold):
    frame = F.to_tensor(frame)
    frame.unsqueeze_(0)
    start = time.time()
    with torch.no_grad():
        pred = model(frame.to(device))
    end = time.time()
    print("Elapset time PyTorch: %fs" % (end - start))

    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]

    if len(pred_t) != 0:
        pred_t = pred_t[-1]
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]

        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        print(pred_class)
        lista = []
        for box in pred_boxes:
            lista.append((int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])))
        pred_boxes = lista

        # cv2.rectangle(frame, (int(box[0][0][0]), int(box[0][0][1])), (int(box[1][0][0]), int(box[1][0][1])), (255, 0, 0), 5)

        return pred_boxes, pred_class
    return None, None


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model
