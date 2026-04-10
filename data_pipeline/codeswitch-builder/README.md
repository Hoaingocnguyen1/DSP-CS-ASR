# CodeSwitch Builder

Synthetic Data Generation Pipeline for Low-Resource Code-Switching ASR.

## 🧠 Kiến trúc 2 Environments (Chuẩn Thesis Production)

Để tránh xung đột thư viện ác liệt (đặc biệt là phiên bản `numpy` giữa `TTS==0.22.0` và `speechbrain/transformers`), dự án này áp dụng kiến trúc **2 Conda Environments riêng biệt**. Đây là best practice trong các Lab nghiên cứu thực tế.

| Environment | Phiên bản `numpy` | Nhiệm vụ chính | Các script sử dụng |
|-------------|-------------------|----------------|--------------------|
| **`cs_tts`** | `1.22.0` | Sinh giọng nói (Text-to-Speech) | `04_synthesize_audio.py` |
| **`cs_asr`** | `1.26.4` | NLP, Xử lý Text & Train ASR | `01`, `02`, `03`, `05`, `06` |

---

## 🛠 Cài đặt (Setup)

Yêu cầu: Đã cài đặt Miniconda hoặc Anaconda.

```bash
# Tạo môi trường 1: TTS (Sinh audio)
conda env create -f env_tts.yml

# Tạo môi trường 2: ASR (Xử lý text, LLM, làm dataset)
# Lưu ý: Mở env_asr.yml để sửa "pytorch-cuda=11.8" thành 12.1 hoặc cpuonly nếu cần.
conda env create -f env_asr.yml
```

*(Hoặc nếu bạn dùng Linux/Mac đã cài `make`, chỉ cần gõ `make all`)*

---

## 🚀 Workflow Chạy Pipeline

### 🥇 Giai đoạn 1: Chuẩn bị Text & Code-Switching (Dùng `cs_asr`)

```bash
conda activate cs_asr

# 1. Trích xuất câu đơn ngữ từ Datasets (VIVOS, People's Speech)
python scripts/01_prepare_mono_text.py

# 2. Dịch câu tiếng Việt -> Anh (Tùy chọn, để làm song ngữ)
python scripts/02_translate.py

# 3. Yêu cầu LLM trộn câu thành Code-Switching + Gán nhãn LID
python scripts/03_generate_cs_text.py
```

### 🥈 Giai đoạn 2: Trọng tâm — Synthesize Audio (Dùng `cs_tts`)
TTS là một thư viện rất nặng và lock nhiều dependencies cũ. Ta đổi sang môi trường TTS để chạy script này.

```bash
conda activate cs_tts

# 4. Sinh ra Audio .wav (16kHz) từ câu Code-Switching 
python scripts/04_synthesize_audio.py
```

### 🥉 Giai đoạn 3: Đóng gói Dataset (Quay lại `cs_asr`)

```bash
conda activate cs_asr

# 5. Lấy thống kê độ dài, từ vựng
python scripts/05_compute_stats.py

# 6. Build file JSON định dạng chuẩn của SpeechBrain
python scripts/06_build_speechbrain_json.py
```

Sau khi chạy xong số `06`, bạn đã có thư mục `outputs/` chứa `train.json`, `valid.json`, `test.json` sẵn sàng đẩy vào `ASR/train.yaml` để train mô hình SpeechBrain!

---

## ❓ FAQ (Tại sao phải 2 envs?)
- **Tại sao không chung 1 env?** `TTS 0.22.0` ép buộc `numpy==1.22.0`. Trong khi đó `datasets`, `transformers` mới nhất cảnh báo hoặc crash nếu `numpy < 1.24`.
- **TTS có dùng để train ASR không?** Không. TTS chỉ dùng MỘT LẦN DUY NHẤT để tạo dữ liệu training nhân tạo `.wav`. Chạy xong là xong. Ta không nên vì 1 tool dùng một lần mà làm hỏng môi trường training chính (`cs_asr`).
