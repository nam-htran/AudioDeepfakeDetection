# ===== main_deploy.py (Hỗ trợ đầy đủ ResNet18, MaxViT & DeiT) =====
import os
import io
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import librosa
import numpy as np
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torchvision import transforms as TV_Transforms

# --- Thiết lập logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title="Phân Loại Âm Thanh Deepfake (Timm Models)",
    description="API sử dụng các model ResNet18, MaxViT và DeiT, tương thích với notebook huấn luyện."
)

# --- Thiết lập thư mục và Template ---
STATIC_DIR = Path("./static")
TEMPLATES_DIR = Path("./templates")
MODELS_DIR = Path("./models")
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# --- Đồng bộ CONFIG (Dựa trên dataclass 'AudioConfig' từ notebook huấn luyện) ---
class C:
    SR: int = 16000
    N_FFT: int = 1024
    HOP_LENGTH: int = 256
    N_MELS: int = 128
    FMIN: float = 0.0
    FMAX: float = 8000.0
    SEGMENT_LENGTH_SECONDS: float = 3.0
    NORM_EPSILON: float = 1e-6
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGENET_TRANSFORM = TV_Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

logger.info(f"Sử dụng thiết bị: {C.DEVICE}")
logger.info(f"Cấu hình Audio (đồng bộ với notebook): SR={C.SR}, N_MELS={C.N_MELS}, Segment={C.SEGMENT_LENGTH_SECONDS}s")


# --- Cấu hình các Model sẽ được tải ---
# QUAN TRỌNG: Đặt tên file checkpoint của bạn vào đây
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    ### THÊM MỚI: Cấu hình cho model DeiT ###
    "deit_tiny": {
        "name": "DeiT Tiny (224x224)",
        "type": "timm_transformer", # Có thể đặt tên type để phân biệt, nhưng logic xử lý vẫn như nhau
        # THAY THẾ TÊN FILE checkpoint của DeiT vào đây
        "local_path": MODELS_DIR / "best_model_DEIT TINY PATCH16 224_250613_190608 (1).pth",
        "params": {
            "model_name": "deit_tiny_patch16_224.fb_in1k", # Tên chính xác từ timm
            "in_chans": 3,
            "num_classes": 2,
            "pretrained": False,
            "img_size_cnn": 224, # Kích thước input cho model này
        }
    },
    "resnet18": {
        "name": "ResNet18 (224x224)",
        "type": "timm_cnn",
        # THAY THẾ TÊN FILE checkpoint của ResNet18 vào đây
        "local_path": MODELS_DIR / "best_model_ResNet18_250611_165052.pth",
        "params": {
            "model_name": "resnet18",
            "in_chans": 3,
            "num_classes": 2,
            "pretrained": False,
            "img_size_cnn": 224,
        }
    },
    "maxvit_nano": {
        "name": "MaxViT Nano (256x256)",
        "type": "timm_cnn",
        # THAY THẾ TÊN FILE checkpoint của MaxViT vào đây
        "local_path": MODELS_DIR / "best_model_MAXVIT_NANO_RW_256_250611_173010.pth",
        "params": {
            "model_name": "maxvit_nano_rw_256.sw_in1k",
            "in_chans": 3,
            "num_classes": 2,
            "pretrained": False,
            "img_size_cnn": 256,
        }
    },
}

# --- Tái tạo các hàm tiền xử lý từ notebook ---

def standardize_audio(waveform: np.ndarray, sr_orig: int) -> np.ndarray:
    y = waveform
    if sr_orig != C.SR:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=C.SR)
    len_target = int(C.SEGMENT_LENGTH_SECONDS * C.SR)
    if len(y) < len_target:
        y = np.pad(y, (0, len_target - len(y)), 'constant')
    else:
        y = y[:len_target]
    return y

def audio_to_melspec(waveform: np.ndarray) -> np.ndarray:
    m_spec = librosa.feature.melspectrogram(
        y=waveform, sr=C.SR, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX
    )
    return librosa.power_to_db(m_spec, ref=np.max)

def preprocess_spectrogram_for_inference(spec_np: np.ndarray, model_key: str) -> torch.Tensor:
    spec_tensor = torch.from_numpy(spec_np).float()
    model_params = MODEL_CONFIGS[model_key]["params"]
    img_size = model_params["img_size_cnn"]
    
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0)
    
    processed_s = F.interpolate(
        spec_tensor.unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear", align_corners=False
    ).squeeze(0)

    s_min, s_max = processed_s.min(), processed_s.max()
    norm_01 = (processed_s - s_min) / (s_max - s_min + C.NORM_EPSILON)
    norm_3c = norm_01.repeat(3, 1, 1)
    processed_s = C.IMAGENET_TRANSFORM(norm_3c)
    
    return processed_s.to(C.DEVICE).unsqueeze(0)

def process_uploaded_audio(audio_bytes: bytes, model_key: str) -> torch.Tensor:
    try:
        waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=True)
        waveform = waveform.T[0]
        std_waveform = standardize_audio(waveform, sr)
        mel_spec = audio_to_melspec(std_waveform)
        input_tensor = preprocess_spectrogram_for_inference(mel_spec, model_key)
        return input_tensor
    except Exception as e:
        logger.error(f"Lỗi preprocess_audio cho model {model_key}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý file audio: {str(e)}")


# --- Tải Models ---
LOADED_MODELS: Dict[str, Optional[nn.Module]] = {}
logger.info("Bắt đầu khởi tạo và tải models...")

for key, config in MODEL_CONFIGS.items():
    local_model_file: Path = config["local_path"]
    if not local_model_file.exists():
        logger.warning(f"File model '{local_model_file.resolve()}' không tìm thấy. Bỏ qua '{config['name']}'. Vui lòng cập nhật đúng đường dẫn trong MODEL_CONFIGS.")
        LOADED_MODELS[key] = None
        continue
    
    try:
        model_params = config["params"].copy()
        model_params.pop("img_size_cnn", None)
        
        logger.info(f"Tạo model timm: {model_params['model_name']}")
        model_instance = timm.create_model(**model_params)
        
        logger.info(f"Đang tải trọng số cho '{config['name']}' từ '{local_model_file.resolve()}'...")
        checkpoint = torch.load(local_model_file, map_location=C.DEVICE)
        
        model_weights = checkpoint.get('model_state_dict', checkpoint)
        model_instance.load_state_dict(model_weights)
        model_instance.to(C.DEVICE).eval()
        LOADED_MODELS[key] = model_instance
        logger.info(f"-> Model '{config['name']}' đã tải thành công.")

    except Exception as e:
        logger.error(f"LỖI khi tải model '{config['name']}': {e}", exc_info=True)
        LOADED_MODELS[key] = None

available_model_keys = [k for k, m in LOADED_MODELS.items() if m is not None]
if not available_model_keys:
    logger.critical("Không có model nào được tải thành công! API có thể không hoạt động. Hãy kiểm tra lại đường dẫn file checkpoint trong MODEL_CONFIGS.")
else:
    logger.info(f"Các model khả dụng: {available_model_keys}")


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    model_options = [{"key": k, "name": v["name"]} for k, v in MODEL_CONFIGS.items() if k in available_model_keys]
    return templates.TemplateResponse("index.html", {"request": request, "model_options": model_options, "selected_models_keys": []})

@app.post("/predict/", response_class=HTMLResponse)
async def predict_from_form(request: Request, file: UploadFile = File(...), selected_models: list[str] = Form([])):
    model_options = [{"key": k, "name": v["name"]} for k, v in MODEL_CONFIGS.items() if k in available_model_keys]
    if not file.filename:
        error = "Vui lòng chọn file âm thanh."
        return templates.TemplateResponse("index.html", {"request": request, "error": error, "model_options": model_options, "selected_models_keys": selected_models}, status_code=400)
    if not selected_models:
        error = "Vui lòng chọn ít nhất một model."
        return templates.TemplateResponse("index.html", {"request": request, "error": error, "filename": file.filename, "model_options": model_options, "selected_models_keys": selected_models}, status_code=400)

    try:
        audio_data = await file.read()
    except Exception as e:
        logger.error(f"Lỗi đọc file upload: {e}", exc_info=True)
        error = f"Không thể đọc file: {file.filename}"
        return templates.TemplateResponse("index.html", {"request": request, "error": error, "model_options": model_options, "selected_models_keys": selected_models}, status_code=400)
        
    results = []
    for model_key in selected_models:
        model_instance = LOADED_MODELS.get(model_key)
        model_display_name = MODEL_CONFIGS.get(model_key, {}).get("name", model_key)
        result_data = {"model_key": model_key, "model_name": model_display_name, "error": None}
        
        if model_instance is None:
            result_data["error"] = f"Model '{model_display_name}' không khả dụng."
        else:
            try:
                input_tensor = process_uploaded_audio(audio_data, model_key)
                
                with torch.no_grad():
                    logits = model_instance(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    prob_real = probabilities[0, 1].item()
                    
                result_data["ket_qua"] = "Thật (Real)" if prob_real > 0.5 else "Giả (Fake)"
                result_data["xac_suat_that"] = prob_real
            except HTTPException as e:
                result_data["error"] = e.detail
            except Exception as e:
                logger.error(f"Lỗi dự đoán model '{model_display_name}': {e}", exc_info=True)
                result_data["error"] = f"Lỗi dự đoán không xác định ({type(e).__name__})"
        results.append(result_data)
        
    # Sắp xếp lại kết quả theo thứ tự trong MODEL_CONFIGS để hiển thị nhất quán
    ordered_results = sorted(results, key=lambda x: list(MODEL_CONFIGS.keys()).index(x['model_key']))
        
    return templates.TemplateResponse("index.html", {"request": request, "predictions": ordered_results, "filename": file.filename, "model_options": model_options, "selected_models_keys": selected_models})

if __name__ == "__main__":
    import uvicorn
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        html_content = """
<!DOCTYPE html><html lang="vi"><head><meta charset="UTF-8"><title>Phân Loại Âm Thanh Deepfake</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet"><style>body{font-family:sans-serif;padding-bottom:70px}.container{max-width:900px}.real{border-left:5px solid green}.fake{border-left:5px solid red}.card-header b{font-size:1.1em}.result-real{color:green;font-weight:bold}.result-fake{color:red;font-weight:bold}.footer{position:fixed;left:0;bottom:0;width:100%;background-color:#f8f9fa;color:#6c757d;text-align:center;padding:10px 0;border-top:1px solid #dee2e6}</style></head><body><div class="container mt-4"><h1 class="text-center mb-4">Phân Loại Âm Thanh Deepfake</h1><p class="text-center text-muted">Sử dụng các model DeiT, ResNet18 và MaxViT.</p>{% if error %}<div class="alert alert-danger">{{ error }}</div>{% endif %}<form action="/predict/" method="post" enctype="multipart/form-data" class="mb-5"><div class="mb-3"><label for="file" class="form-label">Chọn file âm thanh (ví dụ: .wav, .mp3):</label><input type="file" name="file" id="file" class="form-control" required></div>{% if model_options %}<div class="mb-3"><label class="form-label">Chọn Model để dự đoán:</label><div>{% for opt in model_options %}<div class="form-check form-check-inline"><input class="form-check-input" type="checkbox" name="selected_models" value="{{ opt.key }}" id="m_{{ opt.key }}" {% if opt.key in selected_models_keys or not selected_models_keys %}checked{% endif %}><label class="form-check-label" for="m_{{ opt.key }}">{{ opt.name }}</label></div>{% endfor %}</div></div><button type="submit" class="btn btn-primary w-100">Phân Loại</button>{% else %}<div class="alert alert-warning">Không có model nào khả dụng. Vui lòng kiểm tra console server.</div>{% endif %}</form>{% if filename %}<h2 class="mt-4">Kết quả cho file: <span class="fw-normal">{{ filename }}</span></h2>{% endif %}{% if predictions %}<div class="row mt-3 d-flex justify-content-center">{% for pred in predictions %}<div class="col-lg-4 col-md-6 mb-3"><div class="card h-100 {% if pred.error %}border-warning{% elif pred.ket_qua == 'Thật (Real)' %}real{% else %}fake{% endif %}"><div class="card-header"><b>Model: {{ pred.model_name }}</b></div><div class="card-body">{% if pred.error %}<p class="text-danger"><strong>Lỗi:</strong> {{ pred.error }}</p>{% else %}<p><strong>Kết quả:</strong> <span class="{% if pred.ket_qua == 'Thật (Real)' %}result-real{% else %}result-fake{% endif %}">{{ pred.ket_qua }}</span></p><p><strong>Độ tin cậy (Thật):</strong> {{ "%.2f"|format(pred.xac_suat_that * 100) }}%</p><div class="progress"><div class="progress-bar bg-success" role="progressbar" style="width: {{ pred.xac_suat_that * 100 }}%;" aria-valuenow="{{ pred.xac_suat_that * 100 }}" aria-valuemin="0" aria-valuemax="100">{{ "%.0f"|format(pred.xac_suat_that * 100) }}%</div></div>{% endif %}</div></div></div>{% endfor %}</div>{% endif %}</div><div class="footer"><p class="mb-0">© 2024 Deepfake Audio Detector.</p></div><script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script></body></html>
"""
        with open(html_path, "w", encoding="utf-8") as f: f.write(html_content)
        logger.info(f"File HTML mẫu '{html_path}' đã được tạo.")
        
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 7000))
    logger.info(f"Chạy uvicorn server tại http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)