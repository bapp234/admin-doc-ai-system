import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
def apply_profile(config, doc_type: str):
    if doc_type == "scan":
        config.ENABLE_SCANNER_EFFECT = False
        config.ENABLE_DESKEW = False
        config.ENABLE_RED_STAMP = True

    elif doc_type == "camera":
        config.ENABLE_SCANNER_EFFECT = True
        config.ENABLE_DESKEW = True
