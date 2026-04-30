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

find double knee:
python src/find_double_knee_candidates.py
This generates:
outputs/results/double_knee_candidates.csv
outputs/figures/double_knee_candidates_page_1.png

Open the generated candidate images and manually check whether they contain both knees.
In outputs/results/double_knee_candidates.csv, mark the images to remove by writing yes in the remove column.

Then move the confirmed double-knee images out of the training dataset:
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