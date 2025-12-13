# AIoT-Final-Project

## Group 20 - Vision Based- Mobile Assistive Detection System

- Goal of the project is to develop a computer vision system for local assistive operations by utilizing computer vision to support the visually impaired in low-light or poor-visibility conditions

- Use onboard detection hardware to locate other individuals and objects and relay their positions to a GUI on a laptop/mobile device for the user

### Pre-Training Instructions
- The notebook used to manage the YOLO model pretraining from the Ultralytics library is available within `~/Model_Pretraining/YOLO_Model_Training.ipynb`. While all library dependencies are managed within the notebook itself; we recommend using a version of Python 3.12+
- On a similar vein, the YOLO model requires considerable resourse in order to effectively train; thus we recommend running the notebook with access to sufficient compute resources (ie. on-prem GPU or hosted-instance)
    - Recommended Instances: (Colab T4-GPU, A100 GPU w/ High RAM option to optimize training performance)
- The output from each model validation step should produce the F1-Confidence, Precision-Confidence, Recall-Confidence, Precision Recall and Confusion Matrix outputs; the pretrained mode validation outputs are saved under the `~val_res/` library, and the base mode validation outputs are saved under the `~no_res/` library.