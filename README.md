# Real-Time Video Analysis

A comprehensive real-time video analysis project that includes multiple detection features using computer vision and deep learning.

## Features

- **Object Detection**: Detect and classify various objects in real-time
- **Human Detection**: Specialized detection for human presence
- **Traffic/Vehicle Detection**: Monitor and detect vehicles in traffic
- **Emotion Detection**: Analyze and detect human emotions
- **Motion Detection**: Track and analyze movement in video streams

## Prerequisites

- Python 3.8+
- OpenCV
- TensorFlow
- Flask
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adityavar808/real-time-video-analysis.git
cd real-time-video-analysis
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download YOLOv3 weights:
   - Download the weights file from [here](https://pjreddie.com/media/files/yolov3.weights)
   - Place the downloaded `yolov3.weights` file in the `python_Scripts` directory

## Project Structure

```
├── app.py                 # Main Flask application
├── python_Scripts/        # Core detection scripts
│   ├── human_yolov3.py   # Human detection module
│   ├── object_yolov3.py  # Object detection module
│   ├── traffic_yolov3.py # Traffic detection module
│   └── yolov3.cfg        # YOLOv3 configuration
├── static/               # Static files (CSS, JS, images)
└── templates/            # HTML templates
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Choose the detection mode you want to use:
   - Object Detection
   - Human Detection
   - Traffic Detection
   - Emotion Detection
   - Motion Detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv3 for object detection
- OpenCV for computer vision capabilities
- Flask for the web interface

## Contact

Aditya Varshney - adityavarshney808@gmail.com

Project Link: [https://github.com/adityavar808/real-time-video-analysis](https://github.com/adityavar808/real-time-video-analysis)
