import os
import cv2
import json
import csv
import unicodedata
import re
import time
import sys
import gc
import threading
import concurrent.futures
try:
    from loguru import logger
except Exception:
    import logging
    logger = logging.getLogger("run_pipeline")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
import numpy as np
try:
    import torch
except Exception:
    torch = None

from src.modules.preprocessing.pipeline import PreprocessingPipeline
from src.modules.preprocessing.config import PreprocessingConfig
from src.modules.ocr.engine import OCREngine
from src.modules.nlp.classifier import TextClassifier
from src.modules.nlp.extractor import InformationExtractor
from src.utils.hardware import optimize_hardware

INPUT_DIR = "data/raw"
OUTPUT_BASE_DIR = "data/output/classified_docs"
DATASET_CSV = "data/dataset/training_dataset.csv"
# Toggle whether a confident header should trigger an early-exit (skip full-page OCR)
ENABLE_EARLY_EXIT = False
# Data collection mode: build a single CSV for training (text,label)
DATA_COLLECTION_MODE = False
DATA_COLLECTION_CSV = "data/dataset/collected_dataset.csv"
# 'header' to collect header text when available, 'full' to collect full-page text
DATA_COLLECTION_SOURCE = "header"
# If True, include classifier-predicted label in the 'label' column; else leave label empty for manual labeling
DATA_COLLECTION_PRELABEL = False
# When True, treat each input file as a single logical document and only
# collect samples from the first page (page_index == 1). This prevents
# multi-page single-document PDFs from polluting the collection CSV with
# subsequent pages that don't contain headers.
# NOTE: multi-page files are processed page-by-page; data collection will
# collect samples from all pages when enabled.

def init_dataset_file():
    os.makedirs(os.path.dirname(DATASET_CSV), exist_ok=True)
    if not os.path.exists(DATASET_CSV):
        with open(DATASET_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerow(['filename', 'label', 'raw_title', 'clean_text'])


def init_collection_file():
    os.makedirs(os.path.dirname(DATA_COLLECTION_CSV), exist_ok=True)
    if not os.path.exists(DATA_COLLECTION_CSV):
        with open(DATA_COLLECTION_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerow(['text', 'label'])


def slugify(name: str) -> str:
    """Simple slugify to ASCII with underscores for filenames."""
    if not name:
        return "file"
    s = unicodedata.normalize('NFKD', name)
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^A-Za-z0-9_-]+', '_', s)
    return s.strip('_') or 'file'


def _atomic_write_json(data: dict, path: str):
    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def _atomic_save_image(img_array, path: str):
    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)).save(tmp, quality=95)
    except Exception:
        cv2.imwrite(tmp, img_array)
    os.replace(tmp, path)

def main():
    print("=" * 60)
    logger.info("🚀 KHỞI ĐỘNG HỆ THỐNG V3 (MULTI-PAGE PDF & TRACEABILITY)")
    
    use_gpu = optimize_hardware()
    init_dataset_file()
    
    prep_config = PreprocessingConfig()
    prep_pipeline = PreprocessingPipeline(prep_config)
    
    # Note: worker threads will initialize their own heavy objects lazily.
    # Keep light main-thread instances for sanity checks if needed.
    ocr_engine = None
    classifier = None
    extractor = None
    
    valid_exts = {'.png', '.jpg', '.jpeg', '.pdf', '.heic'}
    files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not files:
        logger.warning(f"⚠️ Thư mục {INPUT_DIR} trống.")
        return

    # Thread pool for concurrent page processing (3-4 workers)
    csv_lock = threading.Lock()
    thread_local = threading.local()
    # Use a single worker to avoid RAM thrashing on low-memory hosts
    WORKERS = 1

    # Load heavy models once (singleton) to avoid repeated initialization
    ocr_singleton = OCREngine(use_gpu=use_gpu)
    classifier_singleton = TextClassifier()
    extractor_singleton = InformationExtractor()

    def _init_thread_objects():
        # Initialize heavy objects once per worker thread
        if not getattr(thread_local, 'initialized', False):
            # attach shared singletons to thread-local storage for convenience
            thread_local.ocr_engine = ocr_singleton
            thread_local.classifier = classifier_singleton
            thread_local.extractor = extractor_singleton
            thread_local.initialized = True

    def process_page_task(task, csv_writer, csv_lock):
        try:
            _init_thread_objects()
            ocr = thread_local.ocr_engine
            clf = thread_local.classifier
            ext = thread_local.extractor

            filename = task['filename']
            page_index = task['page_index']
            page_identifier = task['page_identifier']
            img_processed = task['img_processed']
            preprocessing_time_ms = int(task.get('preprocessing_time_ms', 0) or 0)

            # Header-only fast pass using detector pre-check and header boxes
            HEADER_FRACTION = 0.30
            # 1) Detect + normalize orientation on full page (ensures working_img and boxes are valid)
            try:
                pre = ocr.detect_and_normalize(img_processed)
                working_img = pre.get('working_img', img_processed)
                boxes = pre.get('boxes', []) or []
                detection_time_ms = int(pre.get('detection_time_ms', 0) or 0)
                rotated_angle = int(pre.get('rotated_angle', 0) or 0)
            except Exception:
                working_img = img_processed
                boxes = []
                detection_time_ms = 0
                rotated_angle = 0

            h_work = working_img.shape[0]
            header_h = max(24, int(h_work * HEADER_FRACTION))
            header_boxes = []
            for b in boxes:
                try:
                    pts = np.array(b, dtype=np.float32)
                    cy = (pts[0][1] + pts[3][1]) / 2
                    if cy <= header_h:
                        header_boxes.append(b)
                except Exception:
                    continue

            try:
                header_ocr = ocr.recognize_from_boxes(working_img, header_boxes, detection_time_ms=detection_time_ms)
            except Exception:
                header_ocr = {"text": "", "text_lines": [], "confidence": 0.0, "detection_time_ms": detection_time_ms, "recognition_time_ms": 0, "boxes_count": 0, "valid_boxes_count": 0}

            header_text = header_ocr.get("text", "")
            header_text_lines = header_ocr.get("text_lines", [])

            header_nlp_time_ms = 0
            if header_text and header_text.strip():
                cleaned_header = ext.normalize_document_numbers(header_text)
                nlp_start = time.time()
                header_class = clf.predict(cleaned_header)
                header_entities = ext.extract(cleaned_header)
                nlp_end = time.time()
                header_nlp_time_ms = int((nlp_end - nlp_start) * 1000)
            else:
                header_class = {"label": "KHAC", "confidence": 0.0, "raw_title": ""}
                header_entities = {}

            # DATA COLLECTION: write header sample to the collection CSV if requested
            if DATA_COLLECTION_MODE and collection_writer is not None:
                try:
                    if DATA_COLLECTION_SOURCE == 'header' and header_text and header_text.strip():
                        collect_text = cleaned_header.replace('\n', ' ') if cleaned_header else header_text.replace('\n', ' ')
                        collect_label = header_class.get('label', '') if DATA_COLLECTION_PRELABEL else ''
                        with csv_lock:
                            collection_writer.writerow([collect_text, collect_label])
                except Exception:
                    logger.exception('Failed to write header sample to collection CSV')

            hdr_label = header_class.get("label", "KHAC")
            hdr_conf = float(header_class.get("confidence", 0.0) or 0.0)
            # EARLY EXIT if header classifier is highly confident (>0.85)
            if hdr_label not in ("KHAC", "CHUA_PHAN_LOAI") and hdr_conf > 0.85:
                label = hdr_label
                raw_title = header_class.get("raw_title", "")
                if ENABLE_EARLY_EXIT:
                    target_folder = os.path.join(OUTPUT_BASE_DIR, label)
                    os.makedirs(target_folder, exist_ok=True)
                    target_image_path = os.path.join(target_folder, f"{page_identifier}.jpg")
                    try:
                        _atomic_save_image(working_img, target_image_path)
                    except Exception:
                        try:
                            cv2.imwrite(target_image_path, working_img)
                        except Exception:
                            logger.exception("Failed to save page image on early-exit")

                    total_time_ms = preprocessing_time_ms + int(header_ocr.get('detection_time_ms', 0) or 0) + int(header_ocr.get('recognition_time_ms', 0) or 0) + header_nlp_time_ms
                    final_data = {
                        "original_file": filename,
                        "page_index": page_index,
                        "document_type": label,
                        "confidence": header_class.get("confidence", 0.0),
                        "extracted_entities": header_entities,
                        "clean_text_lines": header_text_lines,
                        "cleaned_text": cleaned_header if header_text else "",
                        "boxes_count": int(header_ocr.get('boxes_count', 0) or 0),
                        "valid_boxes_count": int(header_ocr.get('valid_boxes_count', 0) or 0),
                        "early_exit": True,
                        "early_exit_reason": "header_confident",
                        "processing_time_ms": {
                            "preprocessing": preprocessing_time_ms,
                            "ocr_detection": int(header_ocr.get('detection_time_ms', 0) or 0),
                            "ocr_recognition": int(header_ocr.get('recognition_time_ms', 0) or 0),
                            "nlp": header_nlp_time_ms,
                            "total": total_time_ms
                        }
                    }

                    json_path = os.path.join(target_folder, f"{page_identifier}.json")
                    try:
                        _atomic_write_json(final_data, json_path)
                    except Exception:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(final_data, f, ensure_ascii=False, indent=4)

                    with csv_lock:
                        csv_writer.writerow([f"{page_identifier}.jpg", label, raw_title, (cleaned_header if header_text else "").replace('\n', ' ')])

                    logger.success(f"⚡ Early Exit via Header! Label [{label}] -> {page_identifier}")
                    return
                else:
                    logger.info(f"Header matched: Label [{label}] (conf {hdr_conf:.2f}) — early-exit disabled, continuing full-page OCR")

            # Full OCR path - prefer using pre-detected boxes to avoid re-detection
            if boxes:
                ocr_result = ocr.recognize_from_boxes(working_img, boxes, detection_time_ms=detection_time_ms)
            else:
                ocr_result = ocr.extract(img_processed)

            text_content = ocr_result.get("text", "")
            text_lines = ocr_result.get("text_lines", [])

            detection_time_ms = int(ocr_result.get('detection_time_ms', 0) or 0)
            recognition_time_ms = int(ocr_result.get('recognition_time_ms', 0) or 0)
            boxes_count = int(ocr_result.get('boxes_count', 0) or 0)
            valid_boxes_count = int(ocr_result.get('valid_boxes_count', 0) or 0)

            if not text_content:
                total_time_ms = preprocessing_time_ms + detection_time_ms + recognition_time_ms
                logger.info(f"⏱ Page {page_identifier} timings (ms): Preproc={preprocessing_time_ms} | OCR_detect={detection_time_ms} | OCR_recog={recognition_time_ms} | NLP=skipped | Boxes={boxes_count}/{valid_boxes_count} | Total={total_time_ms}")
                return

            cleaned_for_nlp = ext.normalize_document_numbers(text_content)
            nlp_start = time.time()
            doc_class = clf.predict(cleaned_for_nlp)
            entities = ext.extract(cleaned_for_nlp)
            nlp_end = time.time()
            nlp_time_ms = int((nlp_end - nlp_start) * 1000)

            # DATA COLLECTION: write full-page sample when requested (or when header absent and source='header')
            if DATA_COLLECTION_MODE and collection_writer is not None:
                try:
                    write_full = False
                    if DATA_COLLECTION_SOURCE == 'full':
                        write_full = True
                    elif DATA_COLLECTION_SOURCE == 'header':
                        # header_text may be defined earlier; only write full if header was missing
                        if not (header_text and header_text.strip()):
                            write_full = True

                    if write_full:
                        collect_text = cleaned_for_nlp.replace('\n', ' ') if cleaned_for_nlp else text_content.replace('\n', ' ')
                        collect_label = doc_class.get('label', '') if DATA_COLLECTION_PRELABEL else ''
                        with csv_lock:
                            collection_writer.writerow([collect_text, collect_label])
                except Exception:
                    logger.exception('Failed to write full-page sample to collection CSV')

            label = doc_class.get("label", "CHUA_PHAN_LOAI")
            raw_title = doc_class.get("raw_title", "")

            target_folder = os.path.join(OUTPUT_BASE_DIR, label)
            os.makedirs(target_folder, exist_ok=True)
            target_image_path = os.path.join(target_folder, f"{page_identifier}.jpg")
            try:
                _atomic_save_image(working_img if 'working_img' in locals() else img_processed, target_image_path)
            except Exception:
                try:
                    cv2.imwrite(target_image_path, (working_img if 'working_img' in locals() else img_processed))
                except Exception:
                    logger.exception("Failed to save page image")

            total_time_ms = preprocessing_time_ms + detection_time_ms + recognition_time_ms + nlp_time_ms
            final_data = {
                "original_file": filename,
                "page_index": page_index,
                "document_type": label,
                "confidence": doc_class.get("confidence", 0.0),
                "extracted_entities": entities,
                "clean_text_lines": text_lines,
                "cleaned_text": cleaned_for_nlp,
                "boxes_count": boxes_count,
                "valid_boxes_count": valid_boxes_count,
                "processing_time_ms": {
                    "preprocessing": preprocessing_time_ms,
                    "ocr_detection": detection_time_ms,
                    "ocr_recognition": recognition_time_ms,
                    "nlp": nlp_time_ms,
                    "total": total_time_ms
                }
            }

            json_path = os.path.join(target_folder, f"{page_identifier}.json")
            try:
                _atomic_write_json(final_data, json_path)
            except Exception:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, ensure_ascii=False, indent=4)

            with csv_lock:
                csv_writer.writerow([f"{page_identifier}.jpg", label, raw_title, text_content.replace('\n', ' ')])

            logger.info(f"⏱ Page {page_identifier} timings (ms): Preproc={preprocessing_time_ms} | OCR_detect={detection_time_ms} | OCR_recog={recognition_time_ms} | NLP={nlp_time_ms} | Boxes={boxes_count}/{valid_boxes_count} | Total={total_time_ms}")
            logger.success(f"✅ Label [{label}] -> {page_identifier} -> OK")

        except Exception as e:
            logger.error(f"Worker error processing {task.get('page_identifier')}: {e}")
        finally:
            # Aggressively free memory after each page to avoid thrashing
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if 'torch' in globals() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # Initialize collection CSV if requested
    if DATA_COLLECTION_MODE:
        init_collection_file()

    collection_file = None
    collection_writer = None

    with open(DATASET_CSV, 'a', encoding='utf-8-sig', newline='') as dataset_file:
        csv_writer = csv.writer(dataset_file)
        # Open collection CSV writer when in collection mode
        if DATA_COLLECTION_MODE:
            collection_file = open(DATA_COLLECTION_CSV, 'a', encoding='utf-8-sig', newline='')
            collection_writer = csv.writer(collection_file)

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            for filename in files:
                try:
                    file_path = os.path.join(INPUT_DIR, filename)
                    base_name_raw = os.path.splitext(filename)[0]
                    base_name = slugify(base_name_raw)
                    logger.info(f"\n📦 ĐANG XỬ LÝ: {filename}")

                    for prep_result in prep_pipeline.run(file_path, ocr_engine="paddle"):
                        img_processed = prep_result.get("image")
                        page_index = prep_result.get("page_index", 1)
                        page_identifier = f"{base_name}_page_{page_index}"

                        if img_processed is None:
                            continue

                        metadata = prep_result.get('metadata', {}) or {}
                        if metadata.get('quality_warning') == 'blank_page':
                            logger.warning(f"⚠️ Bỏ qua trang trống: {filename} page {page_index}")
                            target_folder = os.path.join(OUTPUT_BASE_DIR, 'KHONG_XAC_DINH')
                            os.makedirs(target_folder, exist_ok=True)
                            target_image_path = os.path.join(target_folder, f"{page_identifier}.jpg")
                            try:
                                _atomic_save_image(img_processed, target_image_path)
                            except Exception:
                                try:
                                    cv2.imwrite(target_image_path, img_processed)
                                except Exception:
                                    logger.error("Failed to save blank-page image")

                            preprocessing_time_ms = int(prep_result.get('processing_time_ms', 0) or 0)
                            final_data = {
                                "original_file": filename,
                                "page_index": page_index,
                                "document_type": "KHONG_XAC_DINH",
                                "confidence": 0.0,
                                "extracted_entities": {},
                                "clean_text_lines": [],
                                "processing_time_ms": {
                                    "preprocessing": preprocessing_time_ms,
                                    "ocr_detection": 0,
                                    "ocr_recognition": 0,
                                    "nlp": 0,
                                    "total": preprocessing_time_ms
                                }
                            }
                            json_path = os.path.join(target_folder, f"{page_identifier}.json")
                            try:
                                _atomic_write_json(final_data, json_path)
                            except Exception:
                                with open(json_path, 'w', encoding='utf-8') as f:
                                    json.dump(final_data, f, ensure_ascii=False, indent=4)
                            with csv_lock:
                                csv_writer.writerow([f"{page_identifier}.jpg", "KHONG_XAC_DINH", "", ""])
                            logger.info(f"⏭️ Bỏ qua trang trống -> {page_identifier}")
                            continue

                        preprocessing_time_ms = int(prep_result.get('processing_time_ms', 0) or 0)

                        task = {
                            'filename': filename,
                            'base_name': base_name,
                            'page_index': page_index,
                            'page_identifier': page_identifier,
                            'img_processed': img_processed,
                            'preprocessing_time_ms': preprocessing_time_ms
                        }

                        fut = executor.submit(process_page_task, task, csv_writer, csv_lock)
                        futures.append(fut)

                    # Wait for all submitted tasks for this file to finish before moving to next file
                    for fut in concurrent.futures.as_completed(futures):
                        try:
                            fut.result()
                        except Exception:
                            logger.exception("Error in page worker")

                    # Free memory after finishing this file
                    try:
                        gc.collect()
                    except Exception:
                        pass
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    # reset futures list for next file
                    futures = []

                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    # continue to next file
                    continue

            # Wait for all submitted tasks to finish and surface exceptions
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception:
                    logger.exception("Error in page worker")
        # Close collection file if opened
        if collection_file is not None:
            try:
                collection_file.close()
            except Exception:
                pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run document processing pipeline')
    parser.add_argument('--collect', action='store_true', help='Enable data collection mode (append to collection CSV)')
    parser.add_argument('--source', choices=['header', 'full'], help='Collection source: header or full')
    parser.add_argument('--prelabel', action='store_true', help='Include predicted label in collection CSV')
    parser.add_argument('--enable-early-exit', dest='enable_early_exit', action='store_true', help='Enable early-exit behavior when header classifier is confident')
    parser.add_argument('--no-early-exit', dest='no_early_exit', action='store_true', help='Explicitly disable early-exit')
    parser.add_argument('--clear', dest='clear', action='store_true', help='Delete collection CSV and clear data/output/ before running')

    args = parser.parse_args()

    if args.collect:
        DATA_COLLECTION_MODE = True
    if args.source:
        DATA_COLLECTION_SOURCE = args.source
    if args.prelabel:
        DATA_COLLECTION_PRELABEL = True
    if args.enable_early_exit:
        ENABLE_EARLY_EXIT = True
    if args.no_early_exit:
        ENABLE_EARLY_EXIT = False
    # If requested, clear collection CSV and output directory before running
    if getattr(args, 'clear', False):
        try:
            # Delete collection CSV if it exists
            if os.path.exists(DATA_COLLECTION_CSV):
                try:
                    os.remove(DATA_COLLECTION_CSV)
                    logger.info(f"Removed collection CSV: {DATA_COLLECTION_CSV}")
                except Exception:
                    logger.exception(f"Failed to remove collection CSV: {DATA_COLLECTION_CSV}")

            # Clear the output directory contents (don't remove the output root itself)
            output_root = os.path.dirname(OUTPUT_BASE_DIR) or 'data/output'
            if os.path.exists(output_root):
                import shutil
                for entry in os.listdir(output_root):
                    path = os.path.join(output_root, entry)
                    try:
                        if os.path.isfile(path) or os.path.islink(path):
                            os.remove(path)
                        elif os.path.isdir(path):
                            shutil.rmtree(path)
                    except Exception:
                        logger.exception(f"Failed to remove output path: {path}")
                logger.info(f"Cleared output directory contents: {output_root}")
        except Exception:
            logger.exception('Cleanup (--clear) failed')

    main()