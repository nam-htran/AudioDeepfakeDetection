# ===== main_deploy.py (Sử dụng model từ file .py export từ notebook) =====
import os
import io
from pathlib import Path
from typing import Dict, Optional, List, Any
import logging

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchaudio.transforms as T

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# --- Thiết lập logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import model custom từ các file Python đã export từ notebook ---
try:
    # CNN_Audio từ notebooks/cnn-trainer.py
    from notebooks.cnn_trainer import CNN_Audio 
    # ViT_Audio từ notebooks/vit-trainer.py
    from notebooks.vit_trainer import ViT_Audio  
    logger.info("Đã import CNN_Audio và ViT_Audio từ các file export từ notebook.")
except ImportError as e:
    logger.error(f"Lỗi import model: {e}. Đảm bảo các file cnn-trainer.py và vit-trainer.py tồn tại trong thư mục notebooks/ và chứa đúng định nghĩa lớp CNN_Audio/ViT_Audio. Đồng thời, kiểm tra các thư viện phụ thuộc (ví dụ: einops cho ViT_Audio).")
    raise ImportError(f"Không thể import model: {e}. Vui lòng kiểm tra file và cài đặt.")
except Exception as e: 
    logger.error(f"Lỗi không xác định khi import model: {e}", exc_info=True)
    raise


# --- FastAPI App ---
app = FastAPI(
    title="Phân Loại Âm Thanh Deepfake (Custom Models from Notebook)",
    description="API sử dụng model custom (CNN_Audio, ViT_Audio) từ notebook huấn luyện."
)

# --- Thiết lập thư mục và Template ---
STATIC_DIR = Path("./static")
TEMPLATES_DIR = Path("./templates")
STATIC_DIR.mkdir(exist_ok=True); TEMPLATES_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Đồng bộ CONFIG (Dựa trên config từ notebook) ---
class C:
    SR = 16000
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128 
    CHUNK_DURATION_S = 3.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE_MODEL_INPUT = 224 
    
    CNN_SMALL_CONV_CHANNELS = [32, 64, 128]
    CNN_SMALL_POOL_AFTER_CONV = [True, True, True]
    CNN_SMALL_FC_UNITS = 192 
    CNN_SMALL_DROPOUT = 0.3 

    CNN_LARGE_CONV_CHANNELS = [64, 128, 256, 512, 512]
    CNN_LARGE_POOL_AFTER_CONV = [True, True, True, True, False]
    CNN_LARGE_FC_UNITS = 192 
    CNN_LARGE_DROPOUT = 0.3

    VIT_SMALL_PATCH_SIZE = 16
    VIT_SMALL_IN_CHANNELS = 1 
    VIT_SMALL_DIM = 128      
    VIT_SMALL_DEPTH = 4      
    VIT_SMALL_HEADS = 4      
    VIT_SMALL_MLP_DIM = 256  
    VIT_SMALL_DROPOUT = 0.1  

    VIT_LARGE_PATCH_SIZE = 16
    VIT_LARGE_IN_CHANNELS = 1
    VIT_LARGE_DIM = 384
    VIT_LARGE_DEPTH = 6
    VIT_LARGE_HEADS = 8
    VIT_LARGE_MLP_DIM = 768
    VIT_LARGE_DROPOUT = 0.1
    
    NORM_EPSILON: float = 1e-6

logger.info(f"Sử dụng thiết bị: {C.DEVICE}")
logger.info(f"Cấu hình Audio: SR={C.SR}, N_FFT={C.N_FFT}, HOP_LENGTH={C.HOP_LENGTH}, N_MELS(gốc)={C.N_MELS}")
logger.info(f"Chunk duration: {C.CHUNK_DURATION_S}s. Model input size: {C.IMG_SIZE_MODEL_INPUT}x{C.IMG_SIZE_MODEL_INPUT}")

_mel_temp_transform = T.MelSpectrogram(
    sample_rate=C.SR, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH, n_mels=C.N_MELS, center=True
).to(C.DEVICE)
_dummy_waveform_3s = torch.randn(1, int(C.CHUNK_DURATION_S * C.SR), device=C.DEVICE)
_dummy_mel_spec = _mel_temp_transform(_dummy_waveform_3s)
ACTUAL_TARGET_SPEC_WIDTH = _dummy_mel_spec.shape[-1]
del _mel_temp_transform, _dummy_waveform_3s, _dummy_mel_spec
logger.info(f"Chiều rộng spectrogram gốc cho chunk {C.CHUNK_DURATION_S}s (N_MELS={C.N_MELS}): {ACTUAL_TARGET_SPEC_WIDTH} frames")

MODEL_BASE_DIR = Path("./results")
MODEL_BASE_DIR.mkdir(exist_ok=True)

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cnn_small": {
        "name": "CNN Small (Notebook, 3s)", "type": "CNN_Audio",
        "local_path": MODEL_BASE_DIR / "CNN" / "best_model_CNN_Small_cnn_3s_dataset_102208.pth",
        "params": {
            "img_size": C.IMG_SIZE_MODEL_INPUT, 
            "in_channels": 1, 
            "num_classes": 2,
            "linear_output_units_1st_fc": C.CNN_SMALL_FC_UNITS,
            "cnn_conv_channels": C.CNN_SMALL_CONV_CHANNELS,
            "cnn_pool_after_conv": C.CNN_SMALL_POOL_AFTER_CONV,
            "dropout": C.CNN_SMALL_DROPOUT
        }
    },
    "cnn_large": {
        "name": "CNN Large (Notebook, 3s)", "type": "CNN_Audio",
        "local_path": MODEL_BASE_DIR / "CNN" / "best_model_CNN_Large_cnn_3s_dataset_114040.pth",
        "params": {
            "img_size": C.IMG_SIZE_MODEL_INPUT, 
            "in_channels": 1, 
            "num_classes": 2,
            "linear_output_units_1st_fc": C.CNN_LARGE_FC_UNITS,
            "cnn_conv_channels": C.CNN_LARGE_CONV_CHANNELS,
            "cnn_pool_after_conv": C.CNN_LARGE_POOL_AFTER_CONV,
            "dropout": C.CNN_LARGE_DROPOUT
        }
    },
    "vit_small": {
        "name": "ViT Small (Notebook, 3s)", "type": "ViT_Audio",
        "local_path": MODEL_BASE_DIR / "ViT" / "best_model_ViT_Small_vit_3s_dataset_040441.pth",
        "params": {
            "img_size": C.IMG_SIZE_MODEL_INPUT, 
            "patch_size": C.VIT_SMALL_PATCH_SIZE,
            "num_classes": 2, 
            "in_channels": C.VIT_SMALL_IN_CHANNELS, 
            "dim": C.VIT_SMALL_DIM,
            "depth": C.VIT_SMALL_DEPTH, 
            "heads": C.VIT_SMALL_HEADS,
            "mlp_dim": C.VIT_SMALL_MLP_DIM, 
            "dropout": C.VIT_SMALL_DROPOUT
        }
    },
    "vit_large": {
        "name": "ViT Large (Notebook, 3s)", "type": "ViT_Audio",
        "local_path": MODEL_BASE_DIR / "ViT" / "best_model_ViT_Large_vit_3s_dataset_044740.pth",
        "params": {
            "img_size": C.IMG_SIZE_MODEL_INPUT, 
            "patch_size": C.VIT_LARGE_PATCH_SIZE,
            "num_classes": 2, 
            "in_channels": C.VIT_LARGE_IN_CHANNELS, 
            "dim": C.VIT_LARGE_DIM,
            "depth": C.VIT_LARGE_DEPTH, 
            "heads": C.VIT_LARGE_HEADS,
            "mlp_dim": C.VIT_LARGE_MLP_DIM, 
            "dropout": C.VIT_LARGE_DROPOUT
        }
    }
}

def create_notebook_model(model_type: str, params: Dict[str, Any]) -> nn.Module:
    if model_type == "CNN_Audio": model = CNN_Audio(**params)
    elif model_type == "ViT_Audio": model = ViT_Audio(**params)
    else: raise ValueError(f"Loại mô hình không hợp lệ: {model_type}.")
    return model

mel_transformer_deploy = T.MelSpectrogram(
    sample_rate=C.SR, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH, n_mels=C.N_MELS, center=True
).to(C.DEVICE)
db_transformer_deploy = T.AmplitudeToDB(stype='power', top_db=80.0).to(C.DEVICE)

def preprocess_audio_for_prediction(audio_bytes: bytes) -> torch.Tensor:
    try:
        waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=True)
        waveform = torch.from_numpy(waveform.T).to(C.DEVICE) # Chuyển lên device ngay sau khi tạo tensor

        if sr != C.SR:
            # Khởi tạo Resample object mà không có device
            resampler = T.Resample(orig_freq=sr, new_freq=C.SR).to(C.DEVICE) # Chuyển resampler lên device
            waveform = resampler(waveform) # waveform đã ở trên C.DEVICE
        
        if waveform.shape[0] > 1: # Nếu là stereo, chuyển thành mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        chunk_samples = int(C.CHUNK_DURATION_S * C.SR)
        num_samples = waveform.shape[1]
        if num_samples > chunk_samples:
            start_idx = (num_samples - chunk_samples) // 2
            waveform_chunked = waveform[:, start_idx : start_idx + chunk_samples]
        elif num_samples < chunk_samples:
            waveform_chunked = torch.nn.functional.pad(waveform, (0, chunk_samples - num_samples), mode='constant', value=0.0)
        else: 
            waveform_chunked = waveform
        
        # Các transformer khác (mel_transformer_deploy, db_transformer_deploy) đã được .to(C.DEVICE) khi khởi tạo
        with torch.no_grad():
            mel_spec_original_dim = mel_transformer_deploy(waveform_chunked) 
            db_spec_original_dim = db_transformer_deploy(mel_spec_original_dim)
            
            if db_spec_original_dim.ndim == 3: 
                db_spec_with_channel = db_spec_original_dim.unsqueeze(1)
            else: 
                db_spec_with_channel = db_spec_original_dim

            resized_spec = F.interpolate(
                db_spec_with_channel,
                size=(C.IMG_SIZE_MODEL_INPUT, C.IMG_SIZE_MODEL_INPUT),
                mode="bilinear", align_corners=False,
            )
            
            mean = resized_spec.mean(dim=(-1, -2), keepdim=True)
            std = resized_spec.std(dim=(-1, -2), keepdim=True) + C.NORM_EPSILON
            input_tensor = (resized_spec - mean) / std
            
            expected_shape = (1, 1, C.IMG_SIZE_MODEL_INPUT, C.IMG_SIZE_MODEL_INPUT)
            if input_tensor.shape != expected_shape:
                logger.error(f"Shape input_tensor cuối cùng: {input_tensor.shape}, expected: {expected_shape}")
                raise ValueError("Lỗi shape tensor tiền xử lý.")
        return input_tensor
    except Exception as e:
        logger.error(f"Lỗi preprocess_audio: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý file: {str(e)}")

LOADED_MODELS: Dict[str, Optional[nn.Module]] = {}
logger.info("Bắt đầu khởi tạo và tải models...")
for model_key, config_entry in MODEL_CONFIGS.items():
    local_model_file: Path = config_entry["local_path"]
    if not local_model_file.exists():
        logger.warning(f"File model '{local_model_file.resolve()}' không tìm thấy. Bỏ qua '{config_entry['name']}'.")
        LOADED_MODELS[model_key] = None
        continue
    try:
        logger.info(f"Khởi tạo model '{config_entry['name']}' (type: {config_entry['type']})...")
        model_instance = create_notebook_model(config_entry["type"], config_entry["params"])
        logger.info(f"Đang tải trọng số cho '{config_entry['name']}' từ '{local_model_file.resolve()}'...")
        checkpoint = torch.load(local_model_file, map_location=C.DEVICE, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_weights = checkpoint["model_state_dict"]
            logger.info(f"Trích xuất 'model_state_dict' cho '{config_entry['name']}'.")
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint: 
            model_weights = checkpoint["state_dict"]
            logger.info(f"Trích xuất 'state_dict' cho '{config_entry['name']}'.")
        else:
            model_weights = checkpoint
            logger.info(f"Checkpoint cho '{config_entry['name']}' là state_dict trực tiếp.")

        if isinstance(model_weights, dict) and any(k.startswith('module.') for k in model_weights.keys()):
            logger.info("Xử lý prefix 'module.' từ DataParallel...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_weights.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model_weights = new_state_dict

        model_instance.load_state_dict(model_weights)
        model_instance.to(C.DEVICE).eval()
        LOADED_MODELS[model_key] = model_instance
        logger.info(f"-> Model '{config_entry['name']}' đã tải thành công.")
    except Exception as e:
        logger.error(f"LỖI khi tải model '{config_entry['name']}': {e}", exc_info=True)
        LOADED_MODELS[model_key] = None

available_model_keys = [key for key, model in LOADED_MODELS.items() if model is not None]
if not available_model_keys: logger.critical("Không có model nào được tải thành công!")
else: logger.info(f"Các model khả dụng: {available_model_keys}")

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    model_options = [{"key": k, "name": v["name"]} for k, v in MODEL_CONFIGS.items() if k in available_model_keys]
    return templates.TemplateResponse("index.html", {"request": request, "model_options": model_options, "selected_models_keys": []})

@app.post("/predict/", response_class=HTMLResponse)
async def predict_from_form(request: Request, file: UploadFile = File(...), selected_models: List[str] = Form([])):
    model_options = [{"key": k, "name": v["name"]} for k, v in MODEL_CONFIGS.items() if k in available_model_keys]
    if not file.filename:
        error = "Vui lòng chọn file âm thanh."
        return templates.TemplateResponse("index.html", {"request": request, "error": error, "model_options": model_options, "selected_models_keys": selected_models}, status_code=400)
    if not selected_models:
        error = "Vui lòng chọn ít nhất một model."
        return templates.TemplateResponse("index.html", {"request": request, "error": error, "filename": file.filename, "model_options": model_options, "selected_models_keys": selected_models}, status_code=400)

    try:
        audio_data = await file.read()
        input_tensor = preprocess_audio_for_prediction(audio_data)
    except HTTPException as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": e.detail, "model_options": model_options, "filename": file.filename, "selected_models_keys": selected_models}, status_code=e.status_code)
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi xử lý file: {e}", exc_info=True)
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Lỗi xử lý file: {type(e).__name__}", "model_options": model_options, "filename": file.filename, "selected_models_keys": selected_models}, status_code=500)

    results = []
    for model_key in selected_models:
        model_instance = LOADED_MODELS.get(model_key)
        model_display_name = MODEL_CONFIGS.get(model_key, {}).get("name", model_key)
        result_data = {"model_key": model_key, "model_name": model_display_name, "error": None}
        if model_instance is None:
            result_data["error"] = f"Model '{model_display_name}' không khả dụng."
        else:
            try:
                with torch.no_grad():
                    output_logits = model_instance(input_tensor)
                    probabilities = torch.softmax(output_logits, dim=1)
                    prob_fake = probabilities[0, 1].item()
                result_data["ket_qua"] = "Giả (Fake)" if prob_fake > 0.5 else "Thật (Real)"
                result_data["xac_suat_gia"] = prob_fake
            except Exception as e:
                logger.error(f"Lỗi dự đoán model '{model_display_name}': {e}", exc_info=True)
                result_data["error"] = f"Lỗi dự đoán ({type(e).__name__})"
        results.append(result_data)
    return templates.TemplateResponse("index.html", {"request": request, "predictions": results, "filename": file.filename, "model_options": model_options, "selected_models_keys": selected_models})

if __name__ == "__main__":
    import uvicorn
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        html_content = """
<!DOCTYPE html><html lang="vi"><head><meta charset="UTF-8"><title>Phân Loại Âm Thanh Deepfake</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet"><style>body{font-family:sans-serif;padding-bottom:70px}.container{max-width:800px}.real{border-left:5px solid green}.fake{border-left:5px solid red}.card-header b{font-size:1.1em}.result-real{color:green;font-weight:bold}.result-fake{color:red;font-weight:bold}.footer{position:fixed;left:0;bottom:0;width:100%;background-color:#f8f9fa;color:#6c757d;text-align:center;padding:10px 0;border-top:1px solid #dee2e6}</style></head><body><div class="container mt-4"><h1 class="text-center mb-4">Phân Loại Âm Thanh Deepfake</h1><p class="text-center text-muted">Sử dụng model custom huấn luyện trên dữ liệu chunked 3 giây.</p>{% if error %}<div class="alert alert-danger">{{ error }}</div>{% endif %}<form action="/predict/" method="post" enctype="multipart/form-data" class="mb-5"><div class="mb-3"><label for="file" class="form-label">Chọn file âm thanh (ví dụ: .wav, .mp3):</label><input type="file" name="file" id="file" class="form-control" required></div>{% if model_options %}<div class="mb-3"><label class="form-label">Chọn Model để dự đoán:</label><div>{% for opt in model_options %}<div class="form-check form-check-inline"><input class="form-check-input" type="checkbox" name="selected_models" value="{{ opt.key }}" id="m_{{ opt.key }}" {% if opt.key in selected_models_keys or not selected_models_keys %}checked{% endif %}><label class="form-check-label" for="m_{{ opt.key }}">{{ opt.name }}</label></div>{% endfor %}</div></div><button type="submit" class="btn btn-primary w-100">Phân Loại</button>{% else %}<div class="alert alert-warning">Không có model nào khả dụng. Vui lòng kiểm tra console server.</div>{% endif %}</form>{% if filename %}<h2 class="mt-4">Kết quả cho file: <span class="fw-normal">{{ filename }}</span></h2>{% endif %}{% if predictions %}<div class="row mt-3">{% for pred in predictions %}<div class="col-md-6 mb-3"><div class="card {% if pred.error %}border-warning{% elif pred.ket_qua == 'Thật (Real)' %}real{% else %}fake{% endif %}"><div class="card-header"><b>Model: {{ pred.model_name }}</b></div><div class="card-body">{% if pred.error %}<p class="text-danger"><strong>Lỗi:</strong> {{ pred.error }}</p>{% else %}<p><strong>Kết quả:</strong> <span class="{% if pred.ket_qua == 'Thật (Real)' %}result-real{% else %}result-fake{% endif %}">{{ pred.ket_qua }}</span></p><p><strong>Xác suất là Giả (Fake):</strong> {{ "%.2f"|format(pred.xac_suat_gia * 100) }}%</p><div class="progress"><div class="progress-bar {% if pred.ket_qua == 'Thật (Real)' %}bg-success{% else %}bg-danger{% endif %}" role="progressbar" style="width: {{ pred.xac_suat_gia * 100 }}%;" aria-valuenow="{{ pred.xac_suat_gia * 100 }}" aria-valuemin="0" aria-valuemax="100">{{ "%.0f"|format(pred.xac_suat_gia * 100) }}%</div>{% if pred.ket_qua == 'Thật (Real)' %}<div class="progress-bar bg-secondary" role="progressbar" style="width: {{ (1-pred.xac_suat_gia) * 100 }}%; opacity:0.3;" aria-valuenow="{{ (1-pred.xac_suat_gia) * 100 }}" aria-valuemin="0" aria-valuemax="100"></div>{% endif %}</div>{% endif %}</div></div></div>{% endfor %}</div>{% endif %}</div><div class="footer"><p class="mb-0">© 2024 Deepfake Audio Detector. Phát triển bởi AI Team.</p></div><script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script></body></html>
"""
        with open(html_path, "w", encoding="utf-8") as f: f.write(html_content)
        logger.info(f"File HTML mẫu '{html_path}' đã được tạo.")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 7000))
    logger.info(f"Chạy uvicorn server tại http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)