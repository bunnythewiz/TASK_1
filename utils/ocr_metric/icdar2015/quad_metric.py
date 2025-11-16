import numpy as np

from .detection.iou import DetectionIoUEvaluator


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class QuadMetric():
    def __init__(self, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator(is_output_polygon=is_output_polygon)

    def measure(self, batch, output, box_thresh=0.6):
        '''
        batch: (image, polygons, ignore_tags)
        '''
        import torch
        
        results = []
        gt_polygons_batch = batch['text_polys']
        ignore_tags_batch = batch['ignore_tags']
        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        
        # Convert tensors to numpy/list
        if isinstance(ignore_tags_batch, torch.Tensor):
            ignore_tags_batch = ignore_tags_batch.cpu().numpy().tolist()
        if isinstance(gt_polygons_batch, torch.Tensor):
            gt_polygons_batch = gt_polygons_batch.cpu().numpy().tolist()
        
        for batch_idx, (polygons, pred_polygons, pred_scores, ignore_tags) in enumerate(
            zip(gt_polygons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch)
        ):
            # Build ground truth annotations with proper shape checking
            if isinstance(polygons, (list, np.ndarray)) and len(polygons) > 0:
                # Ensure ignore_tags has same length as polygons
                if isinstance(ignore_tags, (list, np.ndarray)):
                    if not isinstance(ignore_tags, list):
                        ignore_tags = [ignore_tags]
                    while len(ignore_tags) < len(polygons):
                        ignore_tags.append(ignore_tags[-1] if len(ignore_tags) > 0 else False)
                    
                    gt = []
                    for i in range(len(polygons)):
                        poly_points = np.array(polygons[i], dtype=np.int64)
                        
                        # Ensure shape is [N, 2] for Shapely
                        if poly_points.ndim == 1:
                            # If 1D, reshape to [N/2, 2]
                            poly_points = poly_points.reshape(-1, 2)
                        elif poly_points.ndim == 3:
                            # If 3D (e.g., [1, N, 2]), squeeze
                            poly_points = poly_points.squeeze(0)
                        
                        # Validate shape
                        if poly_points.ndim == 2 and poly_points.shape[1] == 2:
                            gt.append(dict(points=poly_points, ignore=bool(ignore_tags[i])))
                else:
                    # Scalar ignore_tags
                    gt = []
                    for i in range(len(polygons)):
                        poly_points = np.array(polygons[i], dtype=np.int64)
                        if poly_points.ndim == 1:
                            poly_points = poly_points.reshape(-1, 2)
                        elif poly_points.ndim == 3:
                            poly_points = poly_points.squeeze(0)
                        
                        if poly_points.ndim == 2 and poly_points.shape[1] == 2:
                            gt.append(dict(points=poly_points, ignore=bool(ignore_tags)))
            else:
                gt = []
            
            # Build predictions
            if self.is_output_polygon:
                pred = [dict(points=pred_polygons[i]) for i in range(len(pred_polygons))]
            else:
                pred = []
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        pred.append(dict(points=pred_polygons[i, :, :].astype(np.int32)))
            
            results.append(self.evaluator.evaluate_image(gt, pred))
        
        return results

    def validate_measure(self, batch, output, box_thresh=0.6):
        return self.measure(batch, output, box_thresh)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }
