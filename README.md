# Vision Classifiction

## Stack:
- [FastAPI](https://fastapi.tiangolo.com)
- [Python](https://www.python.org)
- [Docker](https://docker.com)
- [Model: Freepik NSFW Image Detector](https://huggingface.co/Freepik/nsfw_image_detector)


## Installation

- For ease of use it's recommended to use the provided [docker-compose.yml](https://github.com/doppeltilde/vision_classification/blob/main/docker-compose.yml).

```yml
services:
  image_video_classification:
    image: ghcr.io/doppeltilde/vision_classification:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface/hub:rw
    environment:
      - DEFAULT_SCORE
      - USE_API_KEYS
      - API_KEYS
    restart: unless-stopped
```

> [!TIP]
> You can find code examples in the [`examples`](./examples/) folder.

## Environment Variables
- Create a `.env` file and set the preferred values.
```sh
DEFAULT_SCORE=0.7

# False == Public Access
# True == Access Only with API Key
USE_API_KEYS=False

# Comma seperated api keys
API_KEYS=abc,123,xyz
```

## Usage

> [!NOTE]
> Please be aware that the initial classification process may require some time, as the model is being downloaded.

> [!TIP]
> Interactive API documentation can be found at: http://localhost:8000/docs

---
_Notice:_ _This project was initally created to be used in-house, as such the
development is first and foremost aligned with the internal requirements._
