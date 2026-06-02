# CNN-based Classification of Knee Osteoarthritis Severity from X-ray Images

## Project Overview

This project aims to classify knee osteoarthritis severity from X-ray images into five Kellgren-Lawrence (KL) grades, from grade 0 to grade 4.

The goal is not only to improve classification accuracy, but also to reduce clinically serious grading mistakes. Since KL grades are ordinal labels, predicting grade 0 as grade 1 is a smaller mistake than predicting grade 0 as grade 4. Therefore, this project focuses on both classification performance and ordinal consistency.

## Problem and Challenges

The task is to predict the KL grade of each knee X-ray image.

| KL Grade | Meaning  |
| -------- | -------- |
| 0        | Normal   |
| 1        | Doubtful |
| 2        | Mild     |
| 3        | Moderate |
| 4        | Severe   |

This task is challenging because:

* neighbouring KL grades can look visually similar;
* early-stage osteoarthritis can be difficult to distinguish;
* the dataset is imbalanced across classes;
* KL grades have a natural ordinal structure;
* distant grading errors are clinically more serious than adjacent errors.

Because of this, accuracy alone is not enough. We also consider ordinal-aware metrics such as MAE, QWK, and distant error analysis.

## Dataset and Preprocessing

The project uses knee X-ray images labelled with KL grades 0 to 4.

Before training, the dataset was cleaned to remove inconsistent images such as double-knee images. This helps keep the input format consistent because the model focuses on one knee region at a time.

The preprocessing pipeline includes:

* converting grayscale X-ray images into 3-channel images;
* resizing images for model input;
* normalizing images using ImageNet mean and standard deviation;
* applying data augmentation only to the training set;
* keeping validation and test data unchanged for fair evaluation.

## Methodology

The final model is based on DenseNet121 pretrained on ImageNet. DenseNet121 was selected as the main backbone because it provides strong feature reuse and stable gradient flow.

The final pipeline includes:

1. image preprocessing;
2. 4-crop input generation;
3. DenseNet121 feature extraction;
4. class-weighted hybrid distance-aware ordinal loss;
5. full fine-tuning;
6. validation and test evaluation.

## Main Improvements

### 1. Ordinal-aware Loss

Standard Cross Entropy treats all classification errors equally. However, KL grades are ordered from 0 to 4, so distant mistakes should be penalized more strongly.

To address this, we used a hybrid distance-aware ordinal loss. This combines standard classification learning with an additional ordinal penalty, helping the model produce more ordinally consistent predictions.

### 2. Class Imbalance Handling

The dataset has an imbalanced class distribution. Some KL grades have more training samples than others, which can bias the model toward majority classes.

To reduce this issue, we tested class imbalance handling methods and selected class-weighted learning for the final model. This gives more importance to minority classes during training.

### 3. 4-Crop Input Strategy

Grad-CAM analysis showed that important osteoarthritis-related features are mainly located around the knee joint region.

Instead of using only the full X-ray image, the final model uses a 4-crop input strategy:

* full image;
* center joint region;
* left joint region;
* right joint region.
![37f255fbc494f2823164e9117372378.png](..%2F..%2F..%2FDocuments%2FWeChat%20Files%2Fwxid_y1lwa02xgvcy22%2FFileStorage%2FTemp%2F37f255fbc494f2823164e9117372378.png)
This helps the model focus on clinically relevant regions such as joint-space narrowing and osteophytes.

### 4. Full Fine-tuning

DenseNet121 was pretrained on ImageNet, but knee X-ray images are very different from natural images. Therefore, we compared different fine-tuning strategies and used full fine-tuning for the final model.

Full fine-tuning allows the whole network to adapt better to X-ray image features.

## Final Model

The final model combines:

* DenseNet121 backbone;
* 4-crop input strategy;
* class-weighted hybrid distance-aware ordinal loss;
* full fine-tuning;
* QWK-based model selection.

The final Kaggle notebook is available in:

```text
final_model/final_model.ipynb
```

## Evaluation

The model is evaluated using multiple metrics:

| Metric         | Purpose                                                        |
| -------------- | -------------------------------------------------------------- |
| Accuracy       | Measures overall classification correctness                    |
| Macro F1       | Measures balanced performance across all classes               |
| MAE            | Measures prediction distance between true and predicted grades |
| QWK            | Measures ordinal agreement                                     |
| Distant Errors | Counts clinically serious large-grade mistakes                 |

The detailed experimental outputs and current metric values are shown in the final model notebook.

## Repository Structure

```text
760-Group1/
├── README.md
├── howtorun.md
├── final_model/
│   └── final_model.ipynb
└── ...
```

## How to Run

The detailed running instructions are provided in:

```text
howtorun.md
```

That file explains how to prepare the environment, attach the dataset in Kaggle, set the dataset path, and run the final model notebook.

## Conclusion

This project shows that knee osteoarthritis severity classification should not rely only on accuracy. Since KL grades are ordinal, reducing distant errors and improving ordinal consistency are also important.

The final approach combines ordinal-aware learning, class imbalance handling, region-focused input, and full fine-tuning to make the model more suitable for KL-grade classification.

## Future Work

Future improvements could include:

* testing on larger and more diverse datasets;
* exploring ensemble methods;
* improving adjacent-grade classification;
* validating the model on external clinical data.
