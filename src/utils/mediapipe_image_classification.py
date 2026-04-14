import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

from PIL import Image

from src.shared.shared import get_model_by_name, get_custom_model


def mediapipe_image_classification(
    img: Image.Image,
    load_from_local_storage: bool = False,
    local_model_path: str = None,
) -> tuple[bool, int, list]:
    try:
        image_classification_model_path = (
            get_custom_model(local_model_path)
            if load_from_local_storage
            else get_model_by_name("Image Classification")
        )

        BaseOptions = mp.tasks.BaseOptions
        ImageClassifier = mp.tasks.vision.ImageClassifier
        ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ImageClassifierOptions(
            base_options=BaseOptions(model_asset_path=image_classification_model_path),
            max_results=5,
            running_mode=VisionRunningMode.IMAGE,
        )
        classifier = ImageClassifier.create_from_options(options)

        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

        results = classifier.classify(mp_image)

        logger.info(f"Classification Results: {results}")

        classification_list = results.classifications if results.classifications else []
        num_classifications = len(classification_list)
        logger.info(f"Classifications {num_classifications}")

        if results.classifications:
            return True, num_classifications, classification_list
        else:
            return False, 0, []

    except Exception as e:
        logger.error(f"Error in face detection: {e}", exc_info=True)
        return False, 0, []
