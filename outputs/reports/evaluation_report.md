# Model Evaluation Report: YOLOv8 Baseline (VAL Split)

## 1. Overall Detection Performance

- **Model Weights**: `best.pt` (6.16 MB)
- **Evaluation Split**: `val`
- **Mean Average Precision (mAP@50)**: **0.0287**
- **Mean Average Precision (mAP@50-95)**: **0.0086**
- **Precision (all classes)**: **0.3692**
- **Recall (all classes)**: **0.1131**

## 2. Real-Time Inference Latency & Throughput

- **Target Hardware**: `cpu`
- **Mean Latency**: **28.78 ms** per frame
- **Median Latency**: **29.19 ms**
- **95th Percentile Latency**: **31.08 ms**
- **Throughput**: **34.7 FPS**

## 3. Per-Class Detection Breakdown

| Defect Category | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| `crazing` | 1.0000 | 0.0000 | 0.0016 | 0.0004 |
| `inclusion` | 0.0281 | 0.1724 | 0.0170 | 0.0064 |
| `patches` | 0.0276 | 0.2700 | 0.0231 | 0.0075 |
| `pitted_surface` | 0.1460 | 0.2195 | 0.1162 | 0.0326 |
| `rolled-in_scale` | 1.0000 | 0.0000 | 0.0133 | 0.0042 |
| `scratches` | 0.0136 | 0.0164 | 0.0011 | 0.0003 |

## 4. Operational Notes
- The evaluation follows COCO standard object detection evaluation protocols.
- When evaluating on `val`, all predictions are compared against ground-truth labels from `datasets/defect_dataset/labels/val`.
- Hyperparameters must NOT be tuned based on the test set.
