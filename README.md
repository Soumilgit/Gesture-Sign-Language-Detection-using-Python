# Gesture Sign Language Detection using Python
Future Scope: To integrate this in a website for the purpose of usability by the deaf/people in need!

# Folder Structure
Here is the folder structure of the repository :
```
📦 
.github
│  └─ FUNDING.yml
LICENSE
README.md
accuracy_score_checker.py
├─ app.py
├─ model
__init__.py
keypoint_classifier
│  │  ├─ keypoint.csv
│  │  ├─ keypoint_classifier.hdf5
│  │  ├─ keypoint_classifier.keras
│  │  ├─ keypoint_classifier.py
│  │  ├─ keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
point_history_classifier
│     ├─ point_history.csv
│     ├─ point_history_classifier.hdf5
│     ├─ point_history_classifier.py
│     ├─ point_history_classifier.tflite
│     └─ point_history_classifier_label.csv
├─ realTime_keypoint_classification.ipynb
├─ realTime_point_history_classification.ipynb
└─ utils
__init__.py
   └─ cvfpscalc.py
```

## Quick Start

### Cloning the Repository
```
git clone https://github.com/Soumilgit/Gesture-Sign-Language-Detection-using-Python.git
cd hand-gesture-recognition-main
```

### Installation

Navigate to the project directory:
```
cd hand-gesture-recognition-main
```

Install the project dependencies using pip:
```
pip install [necessary modules/packages]
```

## Running the Project
## Showcase - please DO CHECK this, if facing issues with running on VSCode:
Start the local server:
```
python app.py

```
## End of MOST IMPORTANT step ABOVE
Accuracy SCORING + LOGGING KEYPOINTS and GESTURE HISTORIES for training CLASSIFIERS:
<p>
<br>Start the local server:</br>

```
python accuracy_score_checker.py

```
</p>

## Deployment Options

### Streamlit

```
cd hand-gesture-recognition-main
```
```
%ls
```
```
!pip install streamlit -q
```
```
!wget -q -O - ipv4.icanhazip.com
```
```
!streamlit run app.py & npx localtunnel --port [port_number]
```
