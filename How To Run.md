## How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/JoyYueLyu/760-Group1.git
cd 760-Group1
```

### 2. Create virtual environment
Windows:
```angular2html
python -m venv .venv
.venv\Scripts\activate
```
macOS / Linux:
```angular2html
python3 -m venv .venv
source .venv/bin/activate
```

### 3.Install dependencies
```angular2html
pip install -r requirements.txt
```
### 4 Download the dataset
https://data.mendeley.com/datasets/t9ndx37v5h/1  
Please download the Digital Knee X-ray Images dataset and place the MedicalExpert-I folder in the following structure:
```angular2html
760-Group1/
├── data/
│   └── raw/
│       └── Digital Knee X-ray Images/
│           └── MedicalExpert-I/
│               ├── 0Normal/
│               ├── 1Doubtful/
│               ├── 2Mild/
│               ├── 3Moderate/
│               └── 4Severe/
```
### 5.Check and clean dataset
```angular2html
python src/check_dataset.py
```
find double knee:  
python src/find_double_knee_candidates.py  
This generates:  
outputs/results/double_knee_candidates.csv  
outputs/figures/double_knee_candidates_page_1.png  

Open the generated candidate images and manually check whether they contain both knees.  
In outputs/results/double_knee_candidates.csv, mark the images to remove by writing yes in the remove column.  

Then move the confirmed double-knee images out of the training dataset:  
```angular2html
python src/remove_double_knee_images.py
python src/check_dataset.py
```

### 6. Create train / validation / test split
```angular2html
python src/create_split.py

This creates:
data/processed/splits.csv
```

### 7.Check DataLoader
```angular2html
python src/dataset.py
```

### 8. Visualize data augmentation
```angular2html
python src/visualize_augmentation.py

outputs/figures/augmentation_examples.png
```

### 9.Train baseline CNN
```angular2html
python src/train_baseline.py

This generates:
outputs/models/baseline_cnn_best.pth
outputs/results/baseline_training_history.csv
outputs/results/baseline_training_summary.csv
outputs/figures/baseline_loss_curve.png
outputs/figures/baseline_accuracy_curve.png
```

### 10.Evaluate baseline CNN
```angular2html
python src/evaluate.py

generates
outputs/results/baseline_test_metrics.csv
outputs/results/baseline_per_class_metrics.csv
outputs/results/baseline_test_predictions.csv
outputs/results/baseline_confusion_matrix.csv
outputs/figures/baseline_confusion_matrix.png
```

### View notebook analysis
```angular2html
notebooks/01_project_progress_baseline_analysis.ipynb
```

### 11. Train ResNet18 transfer learning model
```angular2html
python src/transfer_models/train_resnet18.py

This generates:

outputs/models/resnet18_best.pth
outputs/results/resnet18_training_history.csv
outputs/results/resnet18_training_summary.csv
outputs/figures/resnet18_loss_curve.png
outputs/figures/resnet18_accuracy_curve.png
```

### 12.Evaluate ResNet18 transfer learning model
```angular2html
python src/transfer_models/evaluate_resnet18.py

This generates:
outputs/results/resnet18_test_metrics.csv
outputs/results/resnet18_per_class_metrics.csv
outputs/results/resnet18_test_predictions.csv
outputs/results/resnet18_confusion_matrix.csv
outputs/figures/resnet18_confusion_matrix.png
```
### notebook
```angular2html
notebooks/02_resnet18_transfer_learning_analysis.ipynb
```
###  13. Train ResNet18 with class-weighted loss
```angular2html
python src/transfer_models/train_resnet18_classweighted.py
```
### 14. Evaluate ResNet18 with class-weighted loss
```angular2html
python src/transfer_models/evaluate_resnet18_classweighted.py
```
### 15. Train ResNet18 with ordinal-aware weighted loss
```angular2html
python src/transfer_models/train_resnet18_ordinal.py
```
### 16. Evaluate ResNet18 with ordinal-aware weighted loss
```angular2html
python src/transfer_models/evaluate_resnet18_ordinal.py
```
### 17.grad-CAM
ResNet18  
```angular2html
python src/interpretability/generate_gradcam_resnet18.py --variant standard --samples-per-class 1
```
class-weighted ResNet18  
```angular2html
python src/interpretability/generate_gradcam_resnet18.py --variant classweighted --samples-per-class 1

```
ordinal-aware ResNet18  
```angular2html
python src/interpretability/generate_gradcam_resnet18.py --variant ordinal --samples-per-class 1
```

### 18. Train DenseNet121 transfer learning model
```angular2html
python -m src.transfer_models_DenseNet.train_densenet121

This generates:

outputs/models/densenet121_best.pth
outputs/results/densenet121_training_history.csv
outputs/results/densenet121_training_summary.csv
outputs/figures/densenet121_loss_curve.png
outputs/figures/densenet121_accuracy_curve.png
```

### 19. Evaluate DenseNet121 transfer learning model
```angular2html
python -m src.transfer_models_DenseNet.evaluate_densenet121

This generates:

outputs/results/densenet121_test_metrics.csv
outputs/results/densenet121_per_class_metrics.csv
outputs/results/densenet121_test_predictions.csv
outputs/results/densenet121_confusion_matrix.csv
outputs/figures/densenet121_confusion_matrix.png
```

### 20. Train DenseNet121 with class-weighted loss
```angular2html
python -m src.transfer_models_DenseNet.train_densenet121_classweighted
```

### 21. Evaluate DenseNet121 with class-weighted loss
```angular2html
python -m src.transfer_models_DenseNet.evaluate_densenet121_classweighted
```

### 22. Train DenseNet121 with ordinal-aware weighted loss
```angular2html
python -m src.transfer_models_DenseNet.train_densenet121_ordinal
```

### 23. Evaluate DenseNet121 with ordinal-aware weighted loss
```angular2html
python -m src.transfer_models_DenseNet.evaluate_densenet121_ordinal
```

### 24. grad-CAM
DenseNet121
```angular2html
python -m src.interpretability.generate_gradcam_densenet121 --variant standard --samples-per-class 1
```
class-weighted DenseNet121
```angular2html
python -m src.interpretability.generate_gradcam_densenet121 --variant classweighted --samples-per-class 1
```
ordinal-aware DenseNet121 
```angular2html
python -m src.interpretability.generate_gradcam_densenet121 --variant ordinal --samples-per-class 1
```


## Final Model: How to Run on Kaggle

The final model notebook is provided in:

```text
final_model/final_model.ipynb
```

This notebook was designed to run on Kaggle. Before running the notebook, the pre-split dataset must be uploaded to Kaggle as a dataset.

### 1. Prepare the Dataset

The dataset should already be split into three folders:

```text
split_data/
├── train/
├── val/
└── test/
```

Each folder should contain the five KL grade classes:

```text
0Normal/
1Doubtful/
2Mild/
3Moderate/
4Severe/
```

The expected dataset structure is:

```text
split_data/
├── train/
│   ├── 0Normal/
│   ├── 1Doubtful/
│   ├── 2Mild/
│   ├── 3Moderate/
│   └── 4Severe/
├── val/
│   ├── 0Normal/
│   ├── 1Doubtful/
│   ├── 2Mild/
│   ├── 3Moderate/
│   └── 4Severe/
└── test/
    ├── 0Normal/
    ├── 1Doubtful/
    ├── 2Mild/
    ├── 3Moderate/
    └── 4Severe/
```

### 2. Upload the Dataset to Kaggle

In Kaggle:

1. Open the notebook.
2. Click **Add Input**.
3. Upload or select the pre-split dataset.
4. Make sure the dataset appears in the Kaggle input panel.
5. The dataset should show `train`, `val`, and `test` folders.

### 3. Set the Dataset Path

In the notebook, update the dataset root path if needed:

```python
from pathlib import Path
import os

print("Available Kaggle input folders:")
print(os.listdir("/kaggle/input"))

DATA_ROOT = Path("/kaggle/input/split-data")

if not DATA_ROOT.exists():
    DATA_ROOT = Path("/kaggle/input/split_data")

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

print("DATA_ROOT:", DATA_ROOT)
print("Train:", TRAIN_DIR)
print("Validation:", VAL_DIR)
print("Test:", TEST_DIR)

assert TRAIN_DIR.exists(), f"Train folder not found: {TRAIN_DIR}"
assert VAL_DIR.exists(), f"Validation folder not found: {VAL_DIR}"
assert TEST_DIR.exists(), f"Test folder not found: {TEST_DIR}"
```

If the dataset folder name is different, change this line:

```python
DATA_ROOT = Path("/kaggle/input/split-data")
```

For example:

```python
DATA_ROOT = Path("/kaggle/input/your-dataset-name")
```

### 4. Run the Notebook

After the dataset is attached and the path is correct, run all cells from top to bottom.

The notebook will train and evaluate the final DenseNet121-based model. It reports the main test metrics, including:

* Accuracy
* Macro F1
* Weighted F1
* MAE
* QWK
* Confusion matrix
* Per-class metrics
* Grad-CAM visualisation


