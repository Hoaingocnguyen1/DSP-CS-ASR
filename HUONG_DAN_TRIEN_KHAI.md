# Hướng dẫn Triển khai & Chạy thực nghiệm Mô hình (DSP-CS-ASR)

Tài liệu này hướng dẫn chi tiết các bước từ việc cài đặt môi trường, chuẩn bị dữ liệu, huấn luyện cho đến khi export mô hình và chạy nhận dạng luồng trực tiếp (Streaming Inference) cho repository `DSP-CS-ASR`.

Do repository này đã tích hợp đầy đủ mã nguồn từ thư viện `speechbrain` gốc, mô-đun `DSP-CS-Streaming-ASR` và các công thức huấn luyện (recipes) tùy chỉnh, bạn chỉ cần thực hiện theo các phase dưới đây từ một thư mục gốc.

---

## ⚙️ PHASE 0 — Cài đặt Môi trường
Tất cả các lệnh dưới đây được chạy tại thư mục gốc của repository: `DSP-CS-ASR`

### 0.1 Tạo và Kích hoạt Virtual Environment (Khuyến nghị)
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt (Trên Windows)
.venv\Scripts\activate

# Kích hoạt (Trên Linux/Mac)
source .venv/bin/activate
```

### 0.2 Cài đặt PyTorch
Chọn phiên bản phù hợp với hệ thống GPU của bạn (Yêu cầu Python ≥ 3.10):
```bash
# GPU — CUDA 11.8
pip install torch==2.1.0 torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU — CUDA 12.1
pip install torch==2.1.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 0.3 Cài đặt Repository & Dependencies
Thay vì dùng SpeechBrain từ pip, ta sẽ cài SpeechBrain trực tiếp từ mã nguồn đã chỉnh sửa trong repo này:
```bash
# Yêu cầu cài đặt repo dưới dạng editable
pip install --editable .

# Cài đặt các thư viện bổ trợ cho Code-Switching và Baseline
pip install transformers>=4.38.0 datasets>=2.18.0 sentencepiece>=0.1.99 jiwer soundfile tqdm hyperpyyaml
```

---

## 📦 PHASE 1 — Chuẩn bị Dataset & Tokenizer

Cấu trúc script tiện ích xử lý dữ liệu hiện nằm gọn trong thư mục `DSP-CS-Streaming-ASR/data_pipelines/`.

### 1.1 Tải và Chuẩn bị bộ dữ liệu ViMedCSS
Dữ liệu sẽ được tự động tải về và ép chuẩn sang định dạng JSON cần thiết của SpeechBrain.
```bash
# Chạy script chuẩn bị
python DSP-CS-Streaming-ASR/data_pipelines/codeswitch-builder/scripts/prepare_vimedcss.py \
    --output_dir data/vimedcss
```

### 1.2 Huấn luyện SentencePiece Tokenizer
Bắt buộc phải tạo bộ từ điển BPE phục vụ cho các mô hình trước khi huấn luyện model chính:
```bash
python DSP-CS-Streaming-ASR/data_pipelines/codeswitch-builder/scripts/train_tokenizer.py \
    --train_json data/vimedcss/train.json \
    --output_dir recipes/DSP_CodeSwitch/Tokenizer/save \
    --vocab_size 4000 \
    --model_type bpe
```

---

## 🔬 PHASE 2 — Huấn luyện các mô hình Cơ sở (Baselines)

Các mô hình Baseline nằm tại: `recipes/Baselines/ASR/`
Bạn có thể tùy chọn chạy mô hình Wav2Vec2, XLS-R, HuBERT, WavLM hay Whisper.

Ví dụ huấn luyện với **Wav2Vec2 Vietnamese**:
```bash
cd recipes/Baselines/ASR/
python train_ssl_ctc.py hparams/wav2vec2_vi.yaml

# Hoặc nếu chạy Whisper Small:
# python train_whisper.py hparams/whisper_small.yaml

# Trở lại thư mục gốc sau khi huấn luyện xong
cd ../../../
```

---

## ⭐ PHASE 3 — Huấn luyện Phương pháp Đề xuất (DSP Code-Switch)

Mô hình proposed dùng cơ chế Prompt Injection + Causal GRU LID Head nằm tại: `recipes/DSP_CodeSwitch/ASR/`

### Stage 1: Warmup cho LID Head
Trong giai đoạn này, ta chỉ cập nhật trọng số của nhánh Causal GRU giúp định dạng ngôn ngữ trước.
```bash
cd recipes/DSP_CodeSwitch/ASR/

# Mở tập tin train.yaml và đảm bảo thiết lập: warmup_only: True
python train.py train.yaml --data_folder ../../../data/vimedcss
```
⏳ _Chờ cho độ chính xác LID tiệm cận hoặc cao hơn 85% trên tập Validation rồi dừng._

### Stage 2: Joint Training (ASR + LID + CTC)
```bash
# Sửa tập tin train.yaml và thay đổi: warmup_only: False
python train.py train.yaml --data_folder ../../../data/vimedcss
```
*(Bạn cũng có thể thử nghiệm nghiệm với tham số `--lora_rank 4` hoặc `--lora_rank 8` lúc gọi script tuỳ vào yêu cầu phần cứng).*

---

## 📊 PHASE 4 — Đánh giá (Evaluation) và Đo đạc (Latency)

### 4.1 Đánh giá Lỗi Code-Switching (cs-WER / vi-WER)
Khi kết thúc huấn luyện, mô hình tạo ra bản đánh giá mã kiểm (`hyp.txt` và `ref.txt`). Ta dùng script chạy báo cáo tách biệt Tiếng Việt và CS:
```bash
python DSP-CS-Streaming-ASR/data_pipelines/codeswitch-builder/scripts/evaluate_cs_errors.py \
    --ref recipes/DSP_CodeSwitch/ASR/results/.../test_ref.txt \
    --hyp recipes/DSP_CodeSwitch/ASR/results/.../test_hyp.txt \
    --json_out results/metrics.json
```

### 4.2 Đo độ trễ Streaming (Latency / RTF)
Kiểm thử Real-time Factor (RTF) cho từng kích thước Chunk (ví dụ: 40ms):
```bash
python DSP-CS-Streaming-ASR/speechbrain_recipe/ASR/measure_latency.py \
    --hparams recipes/DSP_CodeSwitch/ASR/train.yaml \
    --wav data/vimedcss/test/wavs/vimedcss_test_00001.wav \
    --chunk_ms 40 \
    --n_trials 100
```

---

## 🚀 PHASE 5 — Xoay vòng Triển khai (Export ONNX & Streaming Demo)

Để tối ưu hóa ứng dụng, mô hình có thể được Export ra chuẩn ONNX (kèm lượng tử hóa INT8) và dùng trong Streaming Inference:

### 5.1 Export ra file ONNX
```bash
python DSP-CS-Streaming-ASR/speechbrain_recipe/ASR/export_onnx.py \
    --hparams recipes/DSP_CodeSwitch/ASR/train.yaml \
    --checkpoint_dir recipes/DSP_CodeSwitch/ASR/results/.../save \
    --onnx_path dsp_model.onnx \
    --quantize
```

### 5.2 Khởi chạy Demo Streaming
Mô phỏng môi trường Live streaming nhận luồng audio từng chunk nhỏ:
```bash
python DSP-CS-Streaming-ASR/speechbrain_recipe/ASR/inference_streaming.py \
    --onnx_model dsp_model_int8.onnx \
    --wav_file data/vimedcss/test/wavs/vimedcss_test_00001.wav \
    --chunk_ms 40
```
