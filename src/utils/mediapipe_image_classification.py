import os
import uuid
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

from PIL import Image, ImageDraw, ImageFont

from src.shared.shared import get_model_by_name, get_custom_model, OUTPUT_DIR

def mediapipe_image_classification(
    img: Image.Image,
    filename: str = None,
    save_cropped: bool = True,
    load_from_local_storage: bool = False,
    local_model_path: str = None,
) -> tuple[bool, int, list, Image.Image]:
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if not filename:
            filename = getattr(img, "filename", None)
            if filename:
                filename = os.path.basename(filename)
            else:
                filename = f"{uuid.uuid4().hex}.png"

        file_path = os.path.join(OUTPUT_DIR, filename)

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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

        results = classifier.classify(mp_image)

        logger.info(f"Classification Results: {results}")

        classification_list = results.classifications if results.classifications else []
        num_classifications = len(classification_list)
        logger.info(f"Classifications {num_classifications}")

        output_img = img.copy()
        draw = ImageDraw.Draw(output_img)

        try:
            font = ImageFont.truetype("assets/Roboto-ExtraBold.ttf", 20)
        except OSError:
            logger.warning("Failed to load font assets/Roboto-ExtraBold.ttf, using default.")
            font = ImageFont.load_default()

        y_offset = 10
        for classification in classification_list:
            for category in classification.categories:
                score_text = f"{category.category_name}: {category.score * 100:.0f}%"
                logger.info(f"Category Info - {score_text}")
                
                if save_cropped:
                    draw.text((10, y_offset), score_text, font=font, fill="white")
                    y_offset += 20

        if save_cropped:
            output_img.save(file_path)
            logger.info(f"Saved result image to {file_path}")

        if results.classifications:
            return True, num_classifications, classification_list, output_img
        else:
            return False, 0, [], output_img

    except Exception as e:
        logger.error(f"Error in image classification: {e}", exc_info=True)
        return False, 0, [], img