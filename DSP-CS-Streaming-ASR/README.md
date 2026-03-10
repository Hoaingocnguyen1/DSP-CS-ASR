# DSP-CS-Streaming-ASR

**Thesis:** *"Nghiên cứu và phát triển mô hình nhận dạng tiếng nói chuyển ngữ (Code-switching) luồng trực tuyến tài nguyên thấp sử dụng cơ chế Acoustic Prompt Injection bán giám sát."*

---

## 🏗 Kiến trúc Tổng quan

```
Audio (16kHz mono, chunk 40ms)
    │
    ▼
┌────────────────────────────────────────┐
│  W2V-BERT-2.0 Backbone + LoRA (PEFT)  │  ← ~0.5% params trainable
│  frozen pretrained weights + LoRA r=16 │
└────────────────┬───────────────────────┘
                 │ adapted_features [B, T, 1024]
     ┌───────────┴────────────┐
     ▼                         ▼
  Causal GRU LID Head      CTC/Seq2Seq Decoder
  [B, T, 3] (VI/EN/SIL)    [B, S, vocab]
     │  
     │ prompt [B, T, 256]
     ▼
  Linear Projection + Tanh(Gate) 
     │  Acoustic Prompt Injection
     └──────────> features += Tanh(gate) * Dropout(proj(prompt))
                  └── Final Features → Decoder

Loss = L_asr + 0.2 × L_lid + 0.3 × L_ctc
```

---

## 📁 Cấu trúc thư mục

```
DSP-CS-Streaming-ASR/
├── README.md                          ← File này
│
├── data_pipelines/
│   ├── codeswitch-builder/
│   │   └── scripts/
│   │       ├── prepare_vimedcss.py    ← [STEP 1] Tải & chuẩn bị dataset ViMedCSS
│   │       ├── train_tokenizer.py     ← [STEP 2] Train SentencePiece tokenizer
│   │       └── evaluate_cs_errors.py ← [FINAL]  Tính WER / cs-WER / vi-WER
│   │
│   └── pseudo_labeler/
│       └── run_whisper_teacher.py     ← Pseudo-label dữ liệu unlabeled (optional)
│
├── speechbrain/
│   └── recipes/
│       ├── DSP_CodeSwitch/ASR/        ← ⭐ Proposed Method
│       │   ├── dsp_model.py           ← Model architecture
│       │   ├── train.py               ← SpeechBrain training recipe
│       │   └── train.yaml             ← Hyperparameters
│       │
│       └── Baselines/ASR/             ← Baseline comparison
│           ├── train_ssl_ctc.py       ← Recipe cho Wav2Vec2 / HuBERT / WavLM / XLS-R
│           ├── train_whisper.py       ← Recipe cho Whisper
│           └── hparams/
│               ├── wav2vec2_vi.yaml
│               ├── xlsr.yaml
│               ├── hubert.yaml
│               ├── wavlm.yaml
│               └── whisper_small.yaml
│
└── speechbrain_recipe/
    └── ASR/
        ├── inference_streaming.py     ← Real-time streaming demo (ONNX)
        ├── export_onnx.py             ← Export model sang ONNX + INT8
        └── measure_latency.py         ← [EVAL] Đo latency ms/chunk và RTF
```

---

## ⚙️ PHASE 0 — Cài đặt Môi trường

### 0.1  Prerequisites

| Tool   | Version             | Kiểm tra           |
| ------ | ------------------- | ------------------ |
| Python | ≥ 3.10              | `python --version` |
| CUDA   | ≥ 11.8 (nếu có GPU) | `nvidia-smi`       |
| Git    | any                 | `git --version`    |

### 0.2  Tạo Virtual Environment

```bash
# Tạo venv
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source .venv/bin/activate
```

### 0.3  Cài PyTorch (chọn đúng CUDA version)

```bash
# GPU — CUDA 11.8
pip install torch==2.1.0 torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU — CUDA 12.1
pip install torch==2.1.0 torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (inference hoặc không có GPU)
pip install torch==2.1.0 torchaudio
```

### 0.4  Cài tất cả Dependencies

```bash
pip install speechbrain>=1.0.0
pip install transformers>=4.38.0
pip install datasets>=2.18.0
pip install sentencepiece>=0.1.99
pip install jiwer>=4.0.0
pip install soundfile>=0.12.1
pip install tqdm
pip install hyperpyyaml
```

Hoặc một lệnh:
```bash
pip install speechbrain>=1.0.0 transformers>=4.38.0 datasets>=2.18.0 sentencepiece jiwer soundfile tqdm hyperpyyaml
```

### 0.5  Verify cài đặt

```bash
python -c "import speechbrain; print('SpeechBrain:', speechbrain.__version__)"
python -c "from speechbrain.nnet.adapters import LoRA; print('LoRA: OK')"
python -c "import sentencepiece; print('SentencePiece: OK')"
python -c "import jiwer; print('jiwer: OK')"
```

---

## 📦 PHASE 1 — Chuẩn bị Dataset

### 1.1  Tải và chuẩn bị ViMedCSS

```bash
# Tải dataset từ HuggingFace, chuẩn hóa audio (16kHz), format JSON SpeechBrain
python data_pipelines/codeswitch-builder/scripts/prepare_vimedcss.py \
    --output_dir data/vimedcss

# (Nếu muốn tạo Pseudo LID Labels cho Stage 1 - cần cài whisper-timestamped)
pip install whisper-timestamped
python data_pipelines/codeswitch-builder/scripts/prepare_vimedcss.py \
    --output_dir data/vimedcss \
    --pseudo_label \
    --model_size base        # tiny / base / small (base là cân bằng tốt)
```

Output:
```
data/vimedcss/
├── train/wavs/*.wav   ← Audio files (16kHz)
├── valid/wavs/*.wav
├── test/wavs/*.wav
├── train.json         ← SpeechBrain manifest
├── valid.json
└── test.json
```

### 1.2  Train SentencePiece Tokenizer

> ⚠️ **Bắt buộc** — Phải chạy TRƯỚC khi train bất kỳ model nào.

```bash
python data_pipelines/codeswitch-builder/scripts/train_tokenizer.py \
    --train_json  data/vimedcss/train.json \
    --output_dir  speechbrain/recipes/DSP_CodeSwitch/Tokenizer/save \
    --vocab_size  4000 \
    --model_type  bpe
```

Output:
```
speechbrain/recipes/DSP_CodeSwitch/Tokenizer/save/
├── tokenizer_4000_bpe.model    ← Dùng trong tất cả CTC baselines
└── tokenizer_4000_bpe.vocab
```

---

## 🔬 PHASE 2 — Train Baseline Models (Table 2)

Chạy thư mục `speechbrain/recipes/Baselines/ASR/`.  
Tất cả models dùng chung tokenizer đã train ở Phase 1.

```bash
cd speechbrain/recipes/Baselines/ASR
```

### Baseline 1: Wav2Vec2 Vietnamese (Nhẹ nhất, khuyên dùng đầu tiên)
```bash
python train_ssl_ctc.py hparams/wav2vec2_vi.yaml
```

### Baseline 2: XLS-R 53 (Multilingual, mạnh nhất trong SSL group)
```bash
# XLS-R lớn → gradient accumulation=8 giúp ổn định training
python train_ssl_ctc.py hparams/xlsr.yaml
```

### Baseline 3: HuBERT
```bash
python train_ssl_ctc.py hparams/hubert.yaml
```

### Baseline 4: WavLM Base+
```bash
python train_ssl_ctc.py hparams/wavlm.yaml
```

### Baseline 5: Whisper Small (Seq2Seq)
```bash
python train_whisper.py hparams/whisper_small.yaml
```

> 💡 **Tip tiết kiệm thời gian:** Chạy Baseline 1 (Wav2Vec2-VI, 95M) trước để kiểm tra pipeline đúng không. Nó train nhanh nhất và dân biết kết quả kỳ vọng.

---

## ⭐ PHASE 3 — Train Proposed Method (Table 3 & 4)

```bash
cd speechbrain/recipes/DSP_CodeSwitch/ASR
```

### Stage 1 — Warmup: Chỉ train Causal GRU LID Head

> Mục tiêu: LID accuracy trên valid > 85% sau 5-10 epochs.

```bash
# Sửa train.yaml: warmup_only: True
python train.py train.yaml \
    --data_folder ../../../data/vimedcss
```

Theo dõi:
```bash
tail -f results/DSP_W2VBERT_CodeSwitch/2025/train_log.txt
```

Dừng khi loss hội tụ. Checkpoint lưu vào `results/.../save/`.

### Stage 2 — Joint Training: ASR + LID + CTC

```bash
# Sửa train.yaml: warmup_only: False
python train.py train.yaml \
    --data_folder ../../../data/vimedcss
```

Loss tổng: `L = L_asr + 0.2 × L_lid + 0.3 × L_ctc`

### Ablation (Table 4) — Chạy với LoRA rank khác nhau

```bash
python train.py train.yaml --lora_rank 4
python train.py train.yaml --lora_rank 8
python train.py train.yaml --lora_rank 16   # default
python train.py train.yaml --lora_rank 32
```

---

## 📊 PHASE 4 — Đánh giá và Ghi kết quả

### 4.1  Tính WER / cs-WER / vi-WER

Sau khi SpeechBrain chạy `evaluate()`, nó xuất ra file `hyp.txt` và `ref.txt`. Dùng:

```bash
python data_pipelines/codeswitch-builder/scripts/evaluate_cs_errors.py \
    --ref  path/to/test_ref.txt \
    --hyp  path/to/test_hyp.txt \
    --json_out results/metrics.json
```

Kết quả mẫu:
```
==================================================
Code-Switching ASR Evaluation Results
==================================================
Total Utterances evaluated: 500
English (CS) Utterances   : 312
Vietnamese Utterances     : 500
--------------------------------------------------
Overall WER        :  18.50 %
Vietnamese (vi-WER):  15.20 %
English (cs-WER)   :  28.40 %  <-- MAIN METRIC
==================================================
```

### 4.2  Đo Latency Streaming (Table 3 — Latency column)

```bash
# Chạy với một file WAV thật
python speechbrain_recipe/ASR/measure_latency.py \
    --hparams speechbrain/recipes/DSP_CodeSwitch/ASR/train.yaml \
    --wav     data/vimedcss/test/wavs/vimedcss_test_00001.wav \
    --chunk_ms 40 \
    --n_trials 100

# Không cần WAV file — dùng synthetic signal
python speechbrain_recipe/ASR/measure_latency.py \
    --chunk_ms 40

# Thử nhiều chunk size (Table 4b)
for ms in 20 40 80 160; do
    python speechbrain_recipe/ASR/measure_latency.py --chunk_ms $ms
done
```

Kết quả mẫu:
```
Chunk latency (mean) : 12.34 ± 1.2 ms
RTF                  : 0.308  ✅ Real-time
```

> Ngưỡng cần đạt: **RTF < 1.0** là đủ real-time. RTF < 0.5 là xuất sắc.

---

## 📦 PHASE 5 — Export & Deployment (Optional)

### 5.1  Export sang ONNX + INT8

```bash
cd speechbrain_recipe/ASR
python export_onnx.py \
    --hparams    ../../speechbrain/recipes/DSP_CodeSwitch/ASR/train.yaml \
    --checkpoint_dir ../../speechbrain/recipes/DSP_CodeSwitch/ASR/results/.../save \
    --onnx_path  dsp_model.onnx \
    --quantize
```

### 5.2  Streaming Inference Demo

```bash
python inference_streaming.py \
    --onnx_model dsp_model_int8.onnx \
    --wav_file   data/vimedcss/test/wavs/vimedcss_test_00001.wav \
    --chunk_ms   40
```

---

## 🐛 Troubleshooting

| Lỗi                                  | Nguyên nhân                                      | Fix                                                             |
| ------------------------------------ | ------------------------------------------------ | --------------------------------------------------------------- |
| `ModuleNotFoundError: speechbrain`   | Chưa cài                                         | `pip install speechbrain>=1.0.0`                                |
| `ModuleNotFoundError: sentencepiece` | Chưa cài                                         | `pip install sentencepiece`                                     |
| `ModuleNotFoundError: jiwer`         | Chưa cài                                         | `pip install jiwer`                                             |
| `CUDA out of memory`                 | Batch size quá lớn                               | Giảm `batch_size: 2` trong yaml                                 |
| `KeyError: dsp_w2vbert`              | `dsp_model.py` không cùng thư mục với `train.py` | Kiểm tra working directory                                      |
| `CTC target length > input length`   | Audio quá ngắn so với transcript                 | Lọc utterance < 1s trong prepare_vimedcss.py                    |
| `tokenizer.model not found`          | Chưa chạy `train_tokenizer.py`                   | Chạy Phase 1.2 trước                                            |
| `HAS_WHISPER_TS not defined`         | Chưa cài `whisper-timestamped`                   | `pip install whisper-timestamped` hoặc bỏ flag `--pseudo_label` |
| `RTF > 1.0`                          | GPU quá chậm hoặc chunk quá nhỏ                  | Tăng `--chunk_ms`, dùng INT8 ONNX                               |

---

## 📋 Checklist Nhanh A-Z

```
□ Phase 0: Cài đặt môi trường + verify imports
□ Phase 1.1: python prepare_vimedcss.py --output_dir data/vimedcss
□ Phase 1.2: python train_tokenizer.py --train_json data/vimedcss/train.json
□ Phase 2: Chạy 5 baselines (điền Table 2)
□ Phase 3 Stage 1: train.yaml warmup_only=True → LID accuracy > 85%
□ Phase 3 Stage 2: train.yaml warmup_only=False → điền Table 3
□ Phase 3 Ablation: lora_rank=4,8,16,32 → điền Table 4a
□ Phase 4.1: evaluate_cs_errors.py → WER / cs-WER / vi-WER
□ Phase 4.2: measure_latency.py chunk=20,40,80,160ms → điền Table 4b
□ Phase 5: Export ONNX + demo streaming (optional)
```

---

## 📚 References

| Module           | Source                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| SpeechBrain      | [speechbrain.github.io](https://speechbrain.github.io)                                                              |
| LoRA             | Hu et al., *ICLR 2022*                                                                                              |
| ViMedCSS Dataset | [tensorxt/ViMedCSS](https://huggingface.co/datasets/tensorxt/ViMedCSS)                                              |
| XLS-R            | [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)                           |
| Whisper          | [openai/whisper-small](https://huggingface.co/openai/whisper-small)                                                 |
| Wav2Vec2-VI      | [nguyenvulebinh/wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) |
