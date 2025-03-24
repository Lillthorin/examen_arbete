from fastapi import FastAPI, UploadFile, File
import os
import shutil
import datetime
import torch
from PIL import Image
import io
import torchvision.transforms as T
from contextlib import asynccontextmanager
from pydantic import BaseModel
#import requests
import gdown

MODEL_DIR = 'models/'
BACKUP_DIR = 'backup_models/'
CURRENT_MODEL = os.path.join(MODEL_DIR, 'current.pt')
MODEL_URL = "https://drive.google.com/uc?export=download&id=1GE9GqnALDDWDqxrEBz-6dbED5B-rehqu"



os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

model = None  # Globalt model-objekt



def download_model():
    url = "https://drive.google.com/uc?id=1GE9GqnALDDWDqxrEBz-6dbED5B-rehqu"
    if not os.path.exists(CURRENT_MODEL):
        print("⏬ Laddar modellen med gdown...")
        gdown.download(url, CURRENT_MODEL, quiet=False)
        print("✅ Modell nedladdad!")



def backup_model():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"model_backup_{timestamp}.pt")
    shutil.copy(CURRENT_MODEL, backup_file)
    print(f"✅ Modell backad upp till {backup_file}")


def rollback_model(backup_filename):
    backup_path = os.path.join(BACKUP_DIR, backup_filename)
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"❌ Backup-filen {backup_path} hittades inte")
    shutil.copy(backup_path, CURRENT_MODEL)
    print(f"✅ Återställde modellen från {backup_path}")

def load_model():
    global model
    if not os.path.exists(CURRENT_MODEL):
        print("❌ Ingen modell att ladda")
        model = None
        return
    print("🔄 Laddar modellen...")
    model = torch.jit.load(CURRENT_MODEL)
    model.eval()
    print("✅ Model loaded and ready!")


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return [transform(image)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    download_model()
    load_model()  # Detta körs när servern startar
    yield

app = FastAPI(lifespan=lifespan)


# 🔄 Ladda upp och uppdatera modellen
@app.post("/update-model/")
async def update_model_endpoint(file: UploadFile = File(...)):
    backup_model()

    if not file.filename.endswith(".pt"):
        return {"error": "File must be a .pt PyTorch model file"}

    model_path = os.path.join(MODEL_DIR, "current.pt")
    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    load_model()

    return {"message": f"✅ Model {file.filename} uploaded and reloaded successfully!", "path": model_path}


# 🔥 Lista tillgängliga backups
@app.get("/list-backups/")
async def list_backups():
    backups = os.listdir(BACKUP_DIR)
    backups.sort(reverse=True)  # Visa senaste först
    return {"backups": backups}




class RollbackRequest(BaseModel):
    backup_file: str

@app.post("/rollback-model/")
async def rollback(rollback_data: RollbackRequest):
    try:
        rollback_model(rollback_data.backup_file)
        load_model()
        return {"message": f"✅ Modellen återställd från {rollback_data.backup_file}"}
    except Exception as e:
        return {"error": str(e)}




# ✅ Exempel: Predict-endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Ingen modell laddad"}

    image_bytes = await file.read()
    image_tensor_list = transform_image(image_bytes)

    with torch.no_grad():
        output = model(image_tensor_list)
        predictions = output[1][0] if isinstance(output, tuple) else output[0]

    if "boxes" not in predictions or len(predictions["boxes"]) == 0:
        return {"message": "No objects detected"}

    result = []
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        result.append({
            "box": box.tolist(),
            "label": int(label),
            "score": float(score)
        })

    return {"predictions": result}

