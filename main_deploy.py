# ===== main_deploy.py (Chạy Localhost - Load Model từ Thư mục Cục bộ) =====
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
import soundfile as sf
# import requests # Không cần requests nữa nếu không tải từ URL
from pathlib import Path
from typing import Dict, Optional, List

# --- FastAPI App ---
app = FastAPI(title="Phân Loại Âm Thanh Deepfake (Local - Models từ Thư mục)", description="API và giao diện phân loại âm thanh thật/giả")

# --- Mount static files và templates ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Config chung ---
SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
MAX_FRAMES_SPEC = 313
FMIN = 0.0
FMAX = None
NORM_EPSILON = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Đường dẫn đến thư mục chứa các model đã huấn luyện ---
MODEL_TRAINED_DIR = Path("./model_trained") # Đảm bảo thư mục này tồn tại và chứa các file .pth

# --- CẤU HÌNH CHO 6 MODELS (Giả định file .pth nằm trong MODEL_TRAINED_DIR) ---
MODEL_CONFIGS = {
    "vit_small": {
        "name": "ViT Small",
        "type": "ViT",
        "local_path": MODEL_TRAINED_DIR / "best_vit_small_model.pth",
        "params": {"patch_size": 16, "embed_dim": 192, "depth": 5, "num_heads": 6, "mlp_ratio": 4.0, "drop_rate": 0.1, "attn_drop_rate": 0.1}
    },
    "vit_medium": {
        "name": "ViT Medium",
        "type": "ViT",
        "local_path": MODEL_TRAINED_DIR / "best_vit_medium_model.pth",
        "params": {"patch_size": 16, "embed_dim": 384, "depth": 6, "num_heads": 6, "mlp_ratio": 4.0, "drop_rate": 0.1, "attn_drop_rate": 0.1}
    },
    "vit_large": {
        "name": "ViT Large",
        "type": "ViT",
        "local_path": MODEL_TRAINED_DIR / "best_vit_large_model.pth",
        "params": {"patch_size": 16, "embed_dim": 512, "depth": 6, "num_heads": 8, "mlp_ratio": 4.0, "drop_rate": 0.1, "attn_drop_rate": 0.1}
    },
    "cnn_small": {
        "name": "CNN Small",
        "type": "CNN",
        "local_path": MODEL_TRAINED_DIR / "best_cnn_small_model.pth",
        "params": {"dropout_rate": 0.4, "channels": [16, 32, 64, 128], "fc_nodes": [128, 32]}
    },
    "cnn_medium": {
        "name": "CNN Medium",
        "type": "CNN",
        "local_path": MODEL_TRAINED_DIR / "best_cnn_medium_model.pth",
        "params": {"dropout_rate": 0.4, "channels": [32, 64, 128, 256], "fc_nodes": [256, 128]}
    },
    "cnn_large": {
        "name": "CNN Large",
        "type": "CNN",
        "local_path": MODEL_TRAINED_DIR / "best_cnn_large_model.pth",
        "params": {"dropout_rate": 0.4, "channels": [32, 64, 128, 256], "fc_nodes": [512, 128]}
    }
}

# --- Định Nghĩa Mô Hình ViT (Giữ nguyên) ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=(N_MELS, MAX_FRAMES_SPEC), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(N_MELS, MAX_FRAMES_SPEC), patch_size=16, in_chans=3, num_classes=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# --- Định Nghĩa Mô Hình CNN (Giữ nguyên) ---
class AudioCNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.4,
                 channels_list=None, fc_nodes_list=None,
                 n_mels=N_MELS, max_frames_spec=MAX_FRAMES_SPEC):
        super(AudioCNN, self).__init__()
        if channels_list is None: channels_list = [32, 64, 128, 256]
        if fc_nodes_list is None: fc_nodes_list = [512, 128]

        self.conv_layers = nn.ModuleList()
        self.bn_conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.drop_conv_layers = nn.ModuleList()
        in_channels = 1
        current_height, current_width = n_mels, max_frames_spec
        for i, out_channels in enumerate(channels_list):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            self.bn_conv_layers.append(nn.BatchNorm2d(out_channels))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2))
            current_dropout_rate = dropout_rate / 2 if i < len(channels_list) / 2 else dropout_rate
            self.drop_conv_layers.append(nn.Dropout2d(current_dropout_rate))
            in_channels = out_channels
            current_height //= 2
            current_width //= 2
        fc_in_features = channels_list[-1] * current_height * current_width
        self.fc_layers = nn.ModuleList()
        self.bn_fc_layers = nn.ModuleList()
        self.drop_fc_layers = nn.ModuleList()
        current_fc_in_dim = fc_in_features
        for i, fc_out_dim in enumerate(fc_nodes_list):
            self.fc_layers.append(nn.Linear(current_fc_in_dim, fc_out_dim))
            self.bn_fc_layers.append(nn.BatchNorm1d(fc_out_dim))
            self.drop_fc_layers.append(nn.Dropout(dropout_rate))
            current_fc_in_dim = fc_out_dim
        self.output_fc = nn.Linear(current_fc_in_dim, num_classes)

    def forward(self, x):
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.bn_conv_layers[i](x)
            x = F.relu(x)
            x = self.pool_layers[i](x)
            x = self.drop_conv_layers[i](x)
        x = x.view(x.size(0), -1)
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
            x = self.bn_fc_layers[i](x)
            x = F.relu(x)
            x = self.drop_fc_layers[i](x)
        x = self.output_fc(x)
        return x

# --- Hàm Tiền Xử Lý Âm Thanh (Giữ nguyên) ---
def audio_to_melspectrogram(audio_data_source, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, max_frames=MAX_FRAMES_SPEC, fmin=FMIN, fmax=FMAX):
    try:
        y, sr_orig = None, sr
        if isinstance(audio_data_source, bytes):
            y, sr_orig = sf.read(io.BytesIO(audio_data_source))
        elif isinstance(audio_data_source, (str, Path)): # Chấp nhận cả str và Path
            if not Path(audio_data_source).exists(): # Kiểm tra file tồn tại
                 raise FileNotFoundError(f"File không tìm thấy: {audio_data_source}")
            y, sr_orig = sf.read(str(audio_data_source)) # sf.read cần string
        else:
            raise ValueError("audio_data_source phải là bytes hoặc đường dẫn file (str/Path).")

        if y.ndim > 1: y = librosa.to_mono(y.T if y.shape[0] > y.shape[1] else y)
        if sr_orig != sr: y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax if fmax is not None else sr/2.0)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        curr_frames = log_mel.shape[1]
        if curr_frames < max_frames:
            pad_val = log_mel.min()
            pad_width = max_frames - curr_frames
            return np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_val)
        elif curr_frames > max_frames:
            return log_mel[:, :max_frames]
        return log_mel
    except FileNotFoundError as e_fnf: # Bắt lỗi FileNotFoundError cụ thể
        raise HTTPException(status_code=404, detail=str(e_fnf))
    except Exception as e:
        # Ghi log chi tiết hơn ở server nếu cần cho việc debug
        # print(f"DEBUG audio_to_melspectrogram error: {type(e).__name__} - {e}")
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý âm thanh: {str(e)}")


# --- Khởi tạo và Tải Tất Cả Models từ Local Path ---
LOADED_MODELS: Dict[str, Optional[nn.Module]] = {}

print("Bắt đầu khởi tạo và tải models từ thư mục cục bộ...")
MODEL_TRAINED_DIR.mkdir(parents=True, exist_ok=True) # Đảm bảo thư mục model_trained tồn tại

for model_key, config in MODEL_CONFIGS.items():
    print(f"\n--- Xử lý model: {config['name']} ({model_key}) ---")
    model_instance = None
    local_model_file = config["local_path"]

    if not local_model_file.exists():
        print(f"CẢNH BÁO: File model {local_model_file} không tìm thấy. Bỏ qua model này.")
        LOADED_MODELS[model_key] = None
        continue # Bỏ qua model này nếu file không tồn tại

    try:
        if config["type"] == "ViT":
            params = config["params"]
            model_instance = VisionTransformer(
                img_size=(N_MELS, MAX_FRAMES_SPEC),
                patch_size=params["patch_size"],
                in_chans=3, num_classes=1,
                embed_dim=params["embed_dim"],
                depth=params["depth"],
                num_heads=params["num_heads"],
                mlp_ratio=params["mlp_ratio"],
                qkv_bias=True,
                drop_rate=params["drop_rate"],
                attn_drop_rate=params["attn_drop_rate"]
            ).to(DEVICE)
        elif config["type"] == "CNN":
            params = config["params"]
            model_instance = AudioCNN(
                num_classes=1,
                dropout_rate=params["dropout_rate"],
                channels_list=params["channels"],
                fc_nodes_list=params["fc_nodes"],
                n_mels=N_MELS,
                max_frames_spec=MAX_FRAMES_SPEC
            ).to(DEVICE)
        
        if model_instance:
            print(f"Đang tải trọng số cho {config['name']} từ {local_model_file}...")
            model_instance.load_state_dict(torch.load(local_model_file, map_location=DEVICE))
            model_instance.eval()
            LOADED_MODELS[model_key] = model_instance
            print(f"Model {config['name']} đã tải và khởi tạo thành công.")
        else:
            print(f"Không thể khởi tạo model structure cho {config['name']}.")
            LOADED_MODELS[model_key] = None

    except Exception as e:
        print(f"Lỗi khi tải hoặc khởi tạo model {config['name']} từ {local_model_file}: {str(e)}")
        LOADED_MODELS[model_key] = None

available_model_keys = [key for key, model in LOADED_MODELS.items() if model is not None]
if not available_model_keys:
    print("\nCẢNH BÁO NGHIÊM TRỌNG: Không có model nào được tải thành công từ thư mục cục bộ!")
    print(f"Vui lòng kiểm tra thư mục '{MODEL_TRAINED_DIR.resolve()}' và đảm bảo các file .pth có tên đúng và hợp lệ.")
else:
    print(f"\nCác model khả dụng đã được tải từ local: {available_model_keys}")


# --- Pydantic Model Cho Response (Giữ nguyên) ---
class PredictionResponse(BaseModel):
    model_key: str
    model_name: str
    ket_qua: str
    xac_suat_gia: float
    error: Optional[str] = None

# --- Trang Chủ (Frontend) (Giữ nguyên, sử dụng model_options từ available_model_keys) ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    model_options = []
    if available_model_keys: # Chỉ tạo options nếu có model khả dụng
        model_options = [{"key": key, "name": MODEL_CONFIGS[key]["name"]} for key in available_model_keys]
    return templates.TemplateResponse("index.html", {"request": request, "model_options": model_options})

# --- API Endpoint Cho Form HTML (Giữ nguyên) ---
@app.post("/predict/", response_class=HTMLResponse)
async def predict_audio_from_form(
    request: Request,
    file: UploadFile = File(...),
    selected_models: List[str] = Form(...)
):
    if not file.filename.endswith((".wav", ".mp3", ".flac")):
        # Lấy lại model_options để hiển thị lại form đúng cách
        model_opts_for_error = [{"key": key, "name": MODEL_CONFIGS[key]["name"]} for key in available_model_keys]
        return templates.TemplateResponse("index.html", {"request": request, "error": "Vui lòng tải lên file WAV, MP3 hoặc FLAC.", "model_options": model_opts_for_error})

    audio_data = await file.read()
    try:
        mel_spec = audio_to_melspectrogram(audio_data)
    except HTTPException as e:
        model_opts_for_error = [{"key": key, "name": MODEL_CONFIGS[key]["name"]} for key in available_model_keys]
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e.detail), "model_options": model_opts_for_error})

    mean = np.mean(mel_spec); std = np.std(mel_spec)
    mel_spec_normalized = (mel_spec - mean) / (std + NORM_EPSILON)
    
    results = []

    if not selected_models:
         model_opts_for_error = [{"key": key, "name": MODEL_CONFIGS[key]["name"]} for key in available_model_keys]
         return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Vui lòng chọn ít nhất một model để dự đoán.",
             "model_options": model_opts_for_error}
        )

    for model_key in selected_models:
        if model_key not in MODEL_CONFIGS:
            results.append(PredictionResponse(
                model_key=model_key, model_name="Không xác định", ket_qua="Lỗi", xac_suat_gia=-1.0,
                error=f"Model key '{model_key}' không hợp lệ."
            ))
            continue

        model_config = MODEL_CONFIGS[model_key]
        model_instance = LOADED_MODELS.get(model_key)

        if model_instance is None:
            results.append(PredictionResponse(
                model_key=model_key, model_name=model_config["name"], ket_qua="Lỗi", xac_suat_gia=-1.0,
                error=f"Model '{model_config['name']}' không khả dụng (chưa được tải hoặc lỗi)."
            ))
            continue
        
        try:
            if model_config["type"] == "ViT":
                mel_spec_input = np.stack([mel_spec_normalized]*3, axis=0)
            else: # CNN
                mel_spec_input = np.expand_dims(mel_spec_normalized, axis=0)
            
            mel_spec_tensor = torch.tensor(mel_spec_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model_instance(mel_spec_tensor)
                prob = torch.sigmoid(output).item()
                prediction_label = "Giả" if prob > 0.5 else "Thật"
                results.append(PredictionResponse(
                    model_key=model_key, model_name=model_config["name"],
                    ket_qua=prediction_label, xac_suat_gia=prob
                ))
        except Exception as e:
            print(f"Lỗi khi dự đoán với model {model_config['name']} ({model_key}): {e}")
            results.append(PredictionResponse(
                model_key=model_key, model_name=model_config["name"], ket_qua="Lỗi", xac_suat_gia=-1.0,
                error=f"Lỗi dự đoán: {str(e)[:100]}..."
            ))

    model_opts_for_display = [{"key": key, "name": MODEL_CONFIGS[key]["name"]} for key in available_model_keys]
    return templates.TemplateResponse("index.html", {
        "request": request, "predictions": results, "filename": file.filename,
        "model_options": model_opts_for_display, "selected_models_keys": selected_models
    })


# --- API Endpoint Cho Client (Giữ nguyên) ---
@app.post("/api/predict/", response_model=List[PredictionResponse])
async def api_predict(
    file: UploadFile = File(...),
    model_keys: List[str] = Query(..., description="Danh sách các model key để dự đoán (vd: vit_small, cnn_large)")
):
    if not file.filename.endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400, detail="Vui lòng tải lên file WAV, MP3 hoặc FLAC.")
    audio_data = await file.read()
    try:
        mel_spec = audio_to_melspectrogram(audio_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi không xác định khi xử lý âm thanh: {str(e)}")

    mean = np.mean(mel_spec); std = np.std(mel_spec)
    mel_spec_normalized = (mel_spec - mean) / (std + NORM_EPSILON)
    
    results = []

    if not model_keys:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp ít nhất một 'model_keys'.")

    for model_key in model_keys:
        if model_key not in MODEL_CONFIGS:
            results.append(PredictionResponse(
                model_key=model_key, model_name="Không xác định", ket_qua="Lỗi", xac_suat_gia=-1.0,
                error=f"Model key '{model_key}' không hợp lệ."
            ))
            continue

        model_config = MODEL_CONFIGS[model_key]
        model_instance = LOADED_MODELS.get(model_key)

        if model_instance is None:
            results.append(PredictionResponse(
                model_key=model_key, model_name=model_config["name"], ket_qua="Lỗi", xac_suat_gia=-1.0,
                error=f"Model '{model_config['name']}' không khả dụng (chưa được tải hoặc lỗi)."
            ))
            continue
        
        try:
            if model_config["type"] == "ViT":
                mel_spec_input = np.stack([mel_spec_normalized]*3, axis=0)
            else: # CNN
                mel_spec_input = np.expand_dims(mel_spec_normalized, axis=0)
            
            mel_spec_tensor = torch.tensor(mel_spec_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model_instance(mel_spec_tensor)
                prob = torch.sigmoid(output).item()
                prediction_label = "Giả" if prob > 0.5 else "Thật"
                results.append(PredictionResponse(
                    model_key=model_key, model_name=model_config["name"],
                    ket_qua=prediction_label, xac_suat_gia=prob
                ))
        except Exception as e:
            print(f"Lỗi API khi dự đoán với model {model_config['name']} ({model_key}): {e}")
            results.append(PredictionResponse(
                model_key=model_key, model_name=model_config["name"], ket_qua="Lỗi", xac_suat_gia=-1.0,
                error=f"Lỗi khi dự đoán với {model_config['name']}: {str(e)[:100]}..."
            ))
            
    return results

# --- Chạy Local Server ---
if __name__ == "__main__":
    import uvicorn
    # Không cần tạo thư mục models_cache nữa vì chúng ta load từ MODEL_TRAINED_DIR
    if not MODEL_TRAINED_DIR.exists() or not MODEL_TRAINED_DIR.is_dir():
        print(f"CẢNH BÁO: Thư mục model huấn luyện '{MODEL_TRAINED_DIR.resolve()}' không tồn tại hoặc không phải là thư mục.")
        print("Các model có thể sẽ không được tải. Vui lòng tạo thư mục và đặt các file .pth vào đó.")
    
    print("Chạy uvicorn server cục bộ trên http://0.0.0.0:8000")
    print("Truy cập http://localhost:8000 để xem giao diện.")
    uvicorn.run(app, host="0.0.0.0", port=8000)