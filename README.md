# COMP702-Image-Processing-and-Computer-Vision-Project
South African Bank Notes Recognition using Image Processing pipeline

## Dataset Setup

### Raw Images
Raw images are included in this repository under `dataset/raw/`.

### Processed Images
Processed and augmented images are available for download here:
(https://drive.google.com/file/d/1S3guDCOKgSFzR33mX_eH2cfstR-eu4eG/view?usp=drive_link)

Download and place the `processed/` folder inside `dataset/`.

### To Regenerate Processed Images Yourself
If you prefer to regenerate locally:
```bash
python setup_project.py
python src/augmentation.py
```
Note: This will take several minutes to complete.

## Preprocessed Images

Preprocessed images are available for download here:
(https://drive.google.com/file/d/1z5A2PwFbIKFTnzTivwg98eDwuTY_OwZS/view?usp=drive_link)

Download and place the `preprocessed/` folder 
inside `dataset/`.

### To Regenerate Preprocessed Images Yourself
```bash
python src/preprocessing.py
```
Note: Requires processed/ images to be present first.

### To Run The Code Yourself
```bash
python -m venv venv/ #windows
#or
python3 -m venv venv/ #mac or linux
#activate venv
pip install -r requirements.txt
python src/knn.py
```
