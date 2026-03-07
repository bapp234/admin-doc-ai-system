from dataclasses import dataclass

@dataclass
class PreprocessingConfig:
    DPI: int = 300

    MAX_IMAGE_SIZE: int = 2500

    ENABLE_DESKEW: bool = True
    DESKEW_MAX_ANGLE: float = 7.0

    ENABLE_CROP: bool = True
    CROP_MIN_AREA_RATIO: float = 0.4

    ENABLE_DENOISE: bool = True
    DENOISE_STRENGTH: int = 10

    ENABLE_BINARIZATION: bool = True

    ENABLE_RED_STAMP: bool = True

    ENABLE_SCANNER_EFFECT: bool = True

