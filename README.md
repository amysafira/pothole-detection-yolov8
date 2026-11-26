# Pothole Detection for Malaysian Roads (YOLOv8)

An end-to-end computer vision project that detects potholes in road photos, rates how urgently they need repair, and turns each scan into a map pin and a downloadable damage report.

**Live app:** [pothole-detector-malaysia.streamlit.app](https://pothole-detector-malaysia.streamlit.app/?embed=true)

---

## Problem

Potholes are a constant complaint on Malaysian roads, and local councils still rely largely on manual inspection and public reports to find them. This project trains an object detection model that locates potholes automatically, so it could run on dashcam images or roadside edge devices to help councils find and prioritise road damage faster.

## Results

Evaluated on a held-out test set of **100 images (288 potholes)** that the model never saw during training.

| Metric | Score |
|---|---|
| **mAP@0.5** | **0.765** |
| mAP@0.5:0.95 | 0.504 |
| Precision | 0.803 |
| Recall | 0.679 |
| Inference speed | ~8.6 ms per image (Tesla T4 GPU) |

In short, about 80% of the model's detections are real potholes, and it finds about 68% of all potholes in the test images.

## Dataset

- **Source:** [Pothole Detection dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/pothole-detection) (andrewmvd)
- **Size:** 665 road images with bounding-box labels, 1 class (`pothole`)
- **Preparation:** converted Pascal VOC (XML) labels to YOLO format and split the data 70 / 15 / 15 into training, validation and test sets with a fixed random seed

## Model and training

- **Model:** YOLOv8s (Ultralytics), fine-tuned from COCO-pretrained weights (transfer learning)
- **Settings:** image size 640, batch size 16, up to 60 epochs with early stopping (patience 15), seed 42
- **Environment:** Google Colab with a free T4 GPU

## App features

The model is deployed as a mobile-friendly Streamlit web app with three tabs.

- **Scan:** take or upload a road photo. Each pothole is boxed and numbered, with its size, confidence and severity (High, Medium or Low).
- **Map:** every scan becomes a pin on a map of the Klang Valley, coloured by repair priority. The app reads GPS from the photo when available, or the user picks the nearest area.
- **History:** a summary of all scans in the session, with totals, a chart by priority and a CSV export.
- **PDF report:** one tap creates a damage report with the photo, pothole details, location and recommended action.

**How severity is estimated:** each pothole's size is measured as a percentage of the photo. Above 3% is High, 1% to 3% is Medium, and below 1% is Low. A road is rated High priority if any pothole is High or there are four or more potholes. This is a simple rule of thumb and should be confirmed by an on-site inspection.

## Repository contents

| File | Purpose |
|---|---|
| `streamlit_app.py` | The web app |
| `best.pt` | Trained YOLOv8s model weights |
| `requirements.txt` | Python packages for the app |
| `packages.txt` | System libraries needed on Streamlit Community Cloud |

## Run it locally

```bash
git clone https://github.com/amysafira/pothole-detection-yolov8.git
cd pothole-detection-yolov8
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Limitations and next steps

- **Recall of 0.68** means some potholes are missed, especially small or shadowed ones. Training on more images, especially photos of Malaysian roads, should help.
- The training data is not from Malaysia, so local road textures, lighting and puddles inside potholes may reduce accuracy.
- Severity is based on size in the photo, not real-world depth or width, which depend on camera distance.
- Possible upgrades: a larger model (YOLOv8m), higher training resolution, and real-time video detection from dashcam footage.

## Author

**Amy Safira binti Muhammad Syahrizal Foo**
Bachelor of Information Systems (Hons.) Intelligent Systems Engineering, Universiti Teknologi MARA (UiTM)

[LinkedIn](https://www.linkedin.com/in/amysafira) · [GitHub](https://github.com/amysafira)
