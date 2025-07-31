# Vision Classification

## Stack:
- [FastAPI](https://fastapi.tiangolo.com)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [Python](https://www.python.org)
- [Docker](https://docker.com)


## Installation

- For ease of use it's recommended to use the provided [docker-compose.yml](https://github.com/doppeltilde/vision_classification/blob/main/docker-compose.yml).

```yml
services:
  vision_classification:
    image: ghcr.io/doppeltilde/vision_classification:latest
    ports:
      - "8000:8000"
    volumes:
      - ./cropped_faces:/app/cropped_faces:rw
      - ./models:/root/.cache/huggingface/hub:rw
    environment:
      - DEFAULT_MODEL_NAME
      - ACCESS_TOKEN
      - DEFAULT_FACE_DETECTION_MODEL_URL
      - USE_API_KEY
      - API_KEY_HASH
      - API_KEY_SALT
      - LOG_LEVEL
    restart: unless-stopped
```

> [!TIP]
> You can find code examples in the [`examples`](./examples/) folder.

## Environment Variables
- Create a `.env` file and set the preferred values.
```sh
DEFAULT_MODEL_NAME=
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

> [!IMPORTANT]
> Set the log level to DEBUG, this will generate an api key, hash, and salt for you.
> Just don't forget to set it back to INFO.

> [!NOTE]
> Please be aware that the initial classification process may require some time, as the model is being downloaded.

> [!TIP]
> Interactive API documentation can be found at: http://localhost:8000/docs

---
_Notice:_ _This project was initally created to be used in-house, as such the
development is first and foremost aligned with the internal requirements._
