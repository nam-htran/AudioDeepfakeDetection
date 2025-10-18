# Deepfake Audio Detection Web App

A web-based application built with FastAPI to detect deepfake audio using state-of-the-art deep learning models. This tool provides a user-friendly interface to upload an audio file and get real-time classification results from multiple models simultaneously.

## Features

- **Multi-Model Analysis**: Utilizes three different powerful models for robust detection:
    - **DeiT (Data-efficient Image Transformer)**
    - **ResNet18**
    - **MaxViT (Multi-Axis Vision Transformer)**
- **User-Friendly Web Interface**: Simple and intuitive UI built with FastAPI, Jinja2, and Bootstrap for easy file uploads and clear result visualization.
- **Side-by-Side Comparison**: Displays predictions from all selected models, allowing for easy comparison of their results and confidence scores.
- **Real-time Processing**: Preprocesses audio on-the-fly, converting it into a Mel Spectrogram and feeding it to the models for instant classification.
- **Extensible Architecture**: Easily add new `timm`-compatible models by simply updating the configuration dictionary in the main script.
- **Handles Common Audio Formats**: Supports various audio formats like `.wav`, `.mp3`, etc., thanks to the `soundfile` and `librosa` libraries.

## How It Works

The application follows a straightforward pipeline from audio upload to classification. The backend processes the audio file, generates a visual representation (Mel Spectrogram), and then uses pre-trained image classification models to determine if the spectrogram belongs to a real or fake audio clip.

<img width="1339" height="794" alt="{EF52B60D-F692-4955-870E-EE4E9CAEDC17}" src="https://github.com/user-attachments/assets/99e8b193-814b-4fce-ba18-b00ec13c958e" />

1.  **Upload**: The user uploads an audio file and selects models via the web interface.
2.  **Preprocessing**: The FastAPI backend receives the file and performs several steps:
    - Resamples the audio to a standard 16,000 Hz.
    - Truncates or pads the audio to a fixed length (3 seconds).
    - Converts the audio waveform into a Mel Spectrogram.
3.  **Tensor Preparation**: The spectrogram is resized to match the model's expected input size (e.g., 224x224), normalized, and converted into a 3-channel tensor, simulating an image.
4.  **Inference**: The prepared tensor is passed to the selected deep learning models (DeiT, ResNet18, MaxViT), which are loaded into memory on server startup.
5.  **Prediction**: Each model outputs a probability score, indicating the likelihood that the audio is "Real".
6.  **Display Results**: The backend sends the predictions back to the user interface, where they are displayed in result cards with clear labels ("Real" or "Fake") and confidence bars.


## Project Structure
.
├── main_deploy.py # Main FastAPI application script
├── models/ # Directory for pre-trained model checkpoints (.pth files)
│ ├── best_model_DEIT TINY PATCH16 224_250613_190608 (1).pth
│ ├── best_model_MAXVIT_NANO_RW_256_250611_173010.pth
│ └── best_model_ResNet18_250611_165052.pth
├── static/
│ └── styles.css # Custom CSS for the frontend
├── templates/
│ └── index.html # Jinja2 template for the web interface
├── manual_dataset/ # (Optional) Sample audio files for testing
│ ├── fake/
│ └── real/
└── README.md # This file

## Setup and Installation

### Prerequisites

- Python 3.8+
- Git

### Steps

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/nam-htran/AudioDeepfakeDetection
    cd AudioDeepfakeDetection
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    fastapi
    uvicorn[standard]
    python-multipart
    jinja2
    torch
    torchvision
    timm
    librosa
    soundfile
    numpy
    ```
    Then, install the packages using pip:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Place the Pre-trained Models:**
    Ensure your trained model checkpoint files (`.pth`) are placed inside the `models/` directory. The application is pre-configured to look for the specific filenames listed in `main_deploy.py`.

5.  **Run the application:**
    ```sh
    uvicorn main_deploy:app --host 0.0.0.0 --port 7000 --reload
    ```
    The `--reload` flag is useful for development as it automatically restarts the server when you make changes to the code.

6.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:7000`.

## Usage

1.  **Open the Web Interface**: Go to `http://127.0.0.1:7000`.
2.  **Upload an Audio File**: Click the upload area to select an audio file (e.g., `.wav`, `.mp3`).
3.  **Select Models**: Check the boxes for the models you want to use for analysis. By default, all available models are selected.
4.  **Classify**: Click the "Phân Loại Ngay" (Classify Now) button.
5.  **View Results**: The page will refresh to show the classification results from each selected model, including the predicted class and confidence scores.

## Technologies Used

- **Backend**: FastAPI, Uvicorn
- **Machine Learning**: PyTorch, Timm
- **Audio Processing**: Librosa, Soundfile, NumPy
- **Frontend**: Jinja2, HTML5, Bootstrap 5
