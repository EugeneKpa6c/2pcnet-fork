from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

import cv2
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from twophase import add_teacher_config
from twophase.engine.trainer import TwoPCTrainer

from twophase.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from twophase.modeling.proposal_generator.rpn import PseudoLabRPN
from twophase.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import twophase.data.datasets.builtin

from twophase.modeling.meta_arch.ts_ensemble import EnsembleTSModel

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes


from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes, pairwise_iou
from detectron2.structures.boxes import BoxMode
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset

import numpy as np

# Словарь для хранения результатов
results = []

def calculate_tp_fp_fn_for_image(pred_boxes_tensor, pred_scores, gt_boxes_list, iou_threshold=0.5):
    device = pred_boxes_tensor.device
    # Убедимся, что pred_boxes уже являются объектом Boxes
    pred_boxes = Boxes(pred_boxes_tensor)

    # Преобразуем gt_boxes_list в тензор и затем в объект Boxes
    gt_boxes_tensor = torch.tensor([BoxMode.convert(box, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for box in gt_boxes_list], device=device)
    gt_boxes = Boxes(gt_boxes_tensor)

    # Отбор предсказаний с вероятностью выше порога
    keep = pred_scores > 0.5
    pred_boxes = pred_boxes[keep]

    # Вычисление IoU и определение TP, FP, FN
    ious = pairwise_iou(pred_boxes, gt_boxes)
    max_ious, _ = ious.max(dim=1)

    tp = (max_ious >= iou_threshold).sum().item()
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    fn = max(0, fn)  # Гарантирует, что FN не будет меньше нуля

    return tp, fp, fn



def perform_inference_and_evaluation(cfg, model):
    model.eval()
    
    
    dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    for dataset_dict in dataset_dicts:
        # Загрузка и предобработка изображения
        image = cv2.imread(dataset_dict["file_name"])
        height, width = image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).cuda()
        inputs = {"image": image, "height": height, "width": width}

        with torch.no_grad():
            outputs = model([inputs])

        # Получение предсказаний для текущего изображения
        pred_boxes = outputs[0]["instances"].pred_boxes.tensor
        pred_scores = outputs[0]["instances"].scores
        gt_boxes_list = [obj["bbox"] for obj in dataset_dict["annotations"]]

        tp, fp, fn = calculate_tp_fp_fn_for_image(pred_boxes, pred_scores, gt_boxes_list)
        print(f"Image ID: {dataset_dict['image_id']}, TP: {tp}, FP: {fp}, FN: {fn}")

        # Расчет метрик
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append((dataset_dict['file_name'], tp, fp, fn, f1_score))


def main(args):
    cfg = get_cfg()
    add_teacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TEST = (args.val_dataset_name,)
    cfg.freeze()
    default_setup(cfg, args)


    if cfg.SEMISUPNET.Trainer == "studentteacher":
        Trainer = TwoPCTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "studentteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    perform_inference_and_evaluation(cfg, model)

    # Сортировка результатов по F1-мере (по убыванию)
    results_sorted = sorted(results, key=lambda x: x[4], reverse=True)

    # Выбор топ 10 лучших и худших результатов
    top_10_best = results_sorted[:20]
    top_10_worst = results_sorted[-20:]

    print("Top 10 Best Results (F1 Score):")
    for result in top_10_best:
        print(f"Image: {os.path.basename(result[0])}, TP: {result[1]}, FP: {result[2]}, FN: {result[3]}, F1: {result[4]:.4f}")

    print("\nTop 10 Worst Results (F1 Score):")
    for result in top_10_worst:
        print(f"Image: {os.path.basename(result[0])}, TP: {result[1]}, FP: {result[2]}, FN: {result[3]}, F1: {result[4]:.4f}")

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.val_dataset_name = "bdd100k_night_val"
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
