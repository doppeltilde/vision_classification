# Vision Classification

## Stack
- [FastAPI](https://fastapi.tiangolo.com)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [Python](https://www.python.org)
- [Docker](https://docker.com)

## Tasks
Depending on the model you choose the following tasks are available:
- Face detection
- Face landmark detection
- Age detection
- Sensitive content detection

A public list of useable models can be found [here](https://huggingface.co/models?pipeline_tag=image-classification&library=onnx,transformers.js&sort=trending).

## Installation

- For ease of use it's recommended to use the provided [compose.yml](https://github.com/doppeltilde/vision_classification/blob/main/compose.yml).

```yml
services:
  vision_classification:
    image: ghcr.io/doppeltilde/vision_classification:latest
    ports:
      - "8000:8000"
    volumes:
      - ./cropped_faces:/app/cropped_faces:rw
      - ./models:/root/.cache/huggingface/hub:rw
      - ./mediapipe_models:/app/mediapipe_models:rw
    env_file:
      - .env
    restart: unless-stopped
```

> [!CAUTION]
> When using [Docker Swarm](https://github.com/doppeltilde/vision_classification/blob/main/compose.swarm.yml), ensure that all necessary volumes are created and accessible before deployment.

> [!TIP]
> You can find code examples in the [`examples`](./examples/) folder.

## Environment Variables
- Create a [`.env`](https://github.com/doppeltilde/vision_classification/blob/main/.env.example) file and set the preferred values.
```sh
# The default model used when no other is set.
DEFAULT_MODEL_NAME=
# Hugging Face access token used to access private models.
ACCESS_TOKEN=
DEFAULT_FACE_DETECTION_MODEL_URL=

# False == Public Access
# True == Access Only with API Key
USE_API_KEY="False"
API_KEY_HASH="<YOUR_GENERATED_KEY_HASH_HERE>"
API_KEY_SALT="<YOUR_GENERATED_SALT_HERE>"

LOG_LEVEL=INFO
```

## Usage

> [!TIP]
> Interactive API documentation can be found at: http://localhost:8000/docs

The API is divided into two distinct categories: Classify Images and MediaPipe Tasks.

#### Classify Images
The **Classify Images** endpoint leverages state-of-the-art models from Hugging Face to perform image classification. It processes input images and returns classification results, including predicted labels and associated confidence scores, based on the selected pre-trained model.

#### Mediapipe tasks
The **MediaPipe Tasks** endpoint utilizes Google's MediaPipe framework to perform various computer vision tasks. It currently exposes the following features:
- **Face Detection**  
  Detects one or more human faces in an image. In addition to returning bounding boxes and detection confidence scores, this task supports:
  - Automatic cropping and saving of detected face regions
  - Extraction and saving of facial landmark coordinates for further processing or analysis

- **Pose Landmark Detection**  
  Identifies and tracks the human body pose by detecting key anatomical landmarks (such as shoulders, elbows, wrists, hips, knees, ankles, etc.). The module returns the coordinates of each landmark along with visibility and presence scores.

> [!IMPORTANT]
> Set the log level to DEBUG, this will generate an api key, hash, and salt for you.
> Just don't forget to set it back to INFO.

> [!NOTE]
> Please be aware that the initial classification process may require some time, as the model is being downloaded.

---
_Notice:_ _This project was initally created to be used in-house, as such the
development is first and foremost aligned with the internal requirements._
