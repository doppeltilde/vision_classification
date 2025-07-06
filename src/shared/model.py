from optimum.pipelines import pipeline
from typing import Optional

classifier: Optional[pipeline] = None

def load_classifier():
    global classifier
    if classifier is None:
        print("Loading image classifier model...")
        classifier = pipeline(
            "image-classification",
            model="onnx-community/nsfw_image_detection-ONNX",
            device=-1,
            accelerator="ort",
        )
        print("Image classifier model loaded.")