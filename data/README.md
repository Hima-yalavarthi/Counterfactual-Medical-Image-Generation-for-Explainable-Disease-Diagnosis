# Dataset Setup Instructions

To download the **Chest X-Ray Pneumonia** dataset from Kaggle, follow these steps:

1.  **Direct Download**:
    - Go to: [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    - Click **Download** (approx. 1.16 GB).
    - Extract the zip file and move the `chest_xray` folder into the `data/` directory of this project.

2.  **Kaggle API (Optional)**:
    If you have the Kaggle API configured, run:
    ```bash
    kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/ --unzip
    ```

### Expected Directory Structure:
Your `data/` folder should look like this:
```text
data/
└── chest_xray/
    ├── train/
    ├── test/
    └── val/
```
Each of these contains `NORMAL` and `PNEUMONIA` subdirectories.
