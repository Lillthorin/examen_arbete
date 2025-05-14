import requests
import os
from pathlib import Path
import json 
import cv2
import numpy as np

"""
Initiering av datasetstruktur om live learning skulle implementeras
-Dataset
--Images/
        -img00001.jpg
        -img00002.jpg
        .........
--Annotations/
        -train.json

"""
DATASET_FOLDER = "dataset"
IMAGES_FOLDER = "dataset/images"
ANNOTATIONS_FOLDER = "dataset/annotations"

ANNOTATION_FILE = "dataset/annotations/train.json"
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)

API_URL = "http://127.0.0.1:8000/"



#  Initialize COCO-style JSON structure
# Används inte i examens arbete
if not os.path.exists(ANNOTATION_FILE) or os.stat(ANNOTATION_FILE).st_size == 0:
    with open(ANNOTATION_FILE, "w") as f:
        json.dump({"categories": [
            {"id": 0, "name": "bkgd", "supercategory": "none"},
            {"id": 1, "name": "box", "supercategory": "bkgd"}
        ], "images": [], "annotations": []}, f)


def predict_no_label(img_array):
    """
    Skickar en numpy-bild till API-servern och returnerar bounding boxes [x1, y1, x2, y2]
    med confidence > 0.80.

    Args:
        img_array (np.ndarray): Bild som numpy-array (t.ex. från kamera)
    
    Returns:
        list: Lista med bounding boxes [[x1, y1, x2, y2], ...]
    """
    result_list = []
    
    try: 
        # Skala till 8-bit om det behövs (t.ex. om Mono16)
        if img_array.dtype != np.uint8:
            img_array = cv2.convertScaleAbs(img_array, alpha=255.0 / np.max(img_array))

    except Exception as e:
        print({e})
        return [],[]
    # Koda till JPEG i minnet
    success, encoded_img = cv2.imencode(".jpg", img_array)
    if not success:
        print(" Kunde inte koda bilden.")
        return [],[]
    predict_url = os.path.join(API_URL, "predict/")
    try:
        response = requests.post(predict_url, files={"file": ("image.jpg", encoded_img.tobytes(), "image/jpeg")})
        data = response.json()

        if data is None or "predictions" not in data:
            return [], []

        # Filtrera predictions med score > 0.70
        high_conf_preds = [pred for pred in data['predictions'] if pred['score'] > 0.50]

        # Extrahera bboxar
        result_list = [[*pred['box']] for pred in high_conf_preds]
        
        return result_list, img_array

    except requests.exceptions.RequestException as e:
        print(" API-fel:", e)
        return [], []
    except json.JSONDecodeError:
        print(" Kunde inte tolka JSON från servern.")
        return [], []

"""
Används inte i examensarbete

Funktioner för att skicka uppdaterad model, lista backup modeller och få prediktioner
samt skapa dataset baserat på de prediktioner modellen gör. Predict_and_label är inte 
ändrad för att ta emot input från blaze-101 kameran.
"""
def predict_and_label(img_array):
    # Hämta bild som numpy-array
    
       # Skala till 8-bit om det behövs (t.ex. om Mono16)
    if img_array.dtype != np.uint8:
        img_array = cv2.convertScaleAbs(img_array, alpha=255.0 / np.max(img_array))
    # Konvertera till JPG-buffer i minnet
    success, encoded_img = cv2.imencode(".jpg", cv2.convertScaleAbs(img_array, alpha=255.0/np.max(img_array)))
    if not success:
        print(" Kunde inte koda bilden.")
        return

    result_list = []
   
   
    # Skicka som multipart/form-data
    response = requests.post(API_URL, files={"file": ("image.jpg", encoded_img.tobytes(), "image/jpeg")})
    data = response.json()

    if data is not None:
        img_decoded = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        h, w, _ = img_decoded.shape
        high_conf_preds = [pred for pred in data['predictions'] if pred['score'] > 0.90]
        
        if not high_conf_preds:
            print("No high confidence predictions. Skipping save.")
            return

        # --- Spara i COCO-format ---
        with open(ANNOTATION_FILE, "r+") as f:
            coco_data = json.load(f)
            image_id = len(coco_data["images"]) + 1
            image_filename = f'{image_id:09d}.jpg'
            image_save_path = os.path.join(IMAGES_FOLDER, image_filename)

            # Spara bilden permanent
            cv2.imwrite(image_save_path, img_decoded)

            coco_data["images"].append({
                "id": image_id,
                "license": 1,
                "file_name": image_filename,
                "height": h,
                "width": w,
                "date_captured": "2024-03-25"
            })

            annotation_id_start = len(coco_data["annotations"])

            for idx, pred in enumerate(high_conf_preds):
                x1, y1, x2, y2 = pred['box']
                width = x2 - x1
                height = y2 - y1
                result_list.append([x1, y1, x2, y2])
                coco_data["annotations"].append({
                    "id": annotation_id_start + idx,
                    "image_id": image_id,
                    "category_id": pred['label'],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "segmentation": [],
                    "iscrowd": 0
                })

            f.seek(0)
            json.dump(coco_data, f, indent=4)
            f.truncate()

        print(f" Bild och annotation sparad: {image_filename}")
        return result_list, img_array

def send_model_update(): 
    with open(r"C:\Users\MathiasTorin\Desktop\Render Mapp\models\current.pt", "rb") as model_file:
        response = requests.post("https://examen-arbete-6011.onrender.com/update-model", files={"file": model_file})

    print(response.json())

def list_models():
    
    response = requests.get("https://examen-arbete-6011.onrender.com/list-backups/")
    print(response.json())

def back_up():
    backup_to_restore = "model_backup_20250324_104943.pt"
    response = requests.post(
        "http://127.0.0.1:8000/rollback-model/",
        json={"backup_file": backup_to_restore}
    )
    print(response.json())






