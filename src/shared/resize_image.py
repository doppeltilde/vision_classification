
from PIL import Image

def resize_image(image: Image.Image, size=(224, 224)) -> Image.Image:
    """
    Resize image to the specified size using high-quality resampling.
    """
    return image.resize(size, Image.Resampling.LANCZOS)