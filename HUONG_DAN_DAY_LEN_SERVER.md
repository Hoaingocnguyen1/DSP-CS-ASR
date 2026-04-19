# Hướng dẫn Đẩy Code và Data lên Server (Vast AI / Custom Server) để Train

Tài liệu này hướng dẫn cách đưa toàn bộ mã nguồn, dữ liệu (dataset) và tokenizer từ máy cá nhân (Local) lên một server từ xa (ví dụ: Vast AI) để huấn luyện mô hình ASR.

## 1. Chuẩn bị thông tin Server

Trước khi bắt đầu, bạn cần thu thập các thông tin kết nối SSH của Server:
- **SSH User:** (ví dụ: `root` - thường dùng trên Vast AI)
- **IP Address:** (ví dụ: `203.0.113.10`)
- **Port:** (ví dụ: `42341` - Vast AI thường cung cấp port ngẫu nhiên, không phải 22)
- **Thư mục đích:** Thư mục chứa project trên server (ví dụ: `/workspace/DSP-CS-ASR`)

---

## 2. Đẩy dữ liệu lên Server (từ máy Local)

Mở terminal trên **máy tính cá nhân của bạn** (đang chứa code ở `/home/hnn/Documents/kltn/DSP-CS-ASR`).

Sử dụng lệnh `rsync` dưới đây để đồng bộ dữ liệu. `rsync` an toàn và có thể tiếp tục (resume) nếu bị rớt mạng.

```bash
# Thay đổi IP (203.0.113.10), Port (42341) và User (root) theo thông tin VastAI của bạn
rsync -avh --progress --partial -e "ssh -p 42341" \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.env' \
  --exclude '__pycache__' \
  --exclude 'results' \
  /home/hnn/Documents/kltn/DSP-CS-ASR/ \
  root@203.0.113.10:/workspace/DSP-CS-ASR/
```

**Giải thích các flag:**
- `--partial`: Rất quan trọng khi dùng Vast AI. Nếu mạng chập chờn, lệnh này sẽ giữ lại các file đang copy dở để lần sau chạy lại không phải copy từ đầu (rất hữu ích cho data lớn).
- `--exclude`: Bỏ qua các thư mục môi trường, git, cache... để giảm dung lượng file cần upload.

---

## 3. SSH vào Server và Cài đặt (trên Server)

Sau khi upload thành công 100%, bạn truy cập vào Terminal của Server:

```bash
# Đổi thông tin Port và IP tương ứng
ssh -p 42341 root@203.0.113.10
```

Vào thư mục dự án trên Server và tiến hành Setup:

```bash
cd /workspace/DSP-CS-ASR

# 0. Cài đặt Miniconda (Chỉ làm bước này NẾU server chưa cài conda)
# Kiểm tra bằng lệnh: conda -V. Nếu báo lỗi "command not found" thì bạn chạy:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda init
# -> Sau đó nhớ reload shell: source ~/.bashrc

# 1. Khôi phục môi trường Conda y hệt như trên máy tính của bạn
# (Dùng file environment.yml vừa được export)
conda env create -f environment.yml

# 2. Kích hoạt môi trường vừa tạo (tên mặc định là asr_env)
conda activate asr_env

# 3. Cập nhật đường dẫn tuyệt đối cho Data trong các file cấu hình YAML
# Cú pháp: ./scripts/fix_data_paths.sh <Đường_dẫn_Local> <Đường_dẫn_Server>
./scripts/fix_data_paths.sh /home/hnn/Documents/kltn/DSP-CS-ASR /workspace/DSP-CS-ASR
```
*(Ghi chú: Đảm bảo các script trong thư mục `scripts/` đã được cấp quyền thực thi bằng lệnh `chmod +x scripts/*.sh`)*

---

## 4. Bắt đầu quá trình Train

Sau khi môi trường đã sẵn sàng, bạn dùng lệnh sau để chạy huấn luyện:

**Để huấn luyện baseline:**
```bash
./scripts/train_server.sh baseline
```

**Để huấn luyện biến thể DSP (Dynamic Soft Prompting):**
```bash
./scripts/train_server.sh dsp
```

**Mẹo nhỏ:** Trên Vast AI hoặc Server remote, bạn nên chạy code trong `tmux` hoặc sử dụng `nohup` để tránh việc train bị dừng nếu bạn bị đứt kết nối SSH đột ngột.
Ví dụ: `nohup ./scripts/train_server.sh dsp > train.log 2>&1 &`

---

## 5. Theo dõi và Đồng bộ Kết quả (Sync Results)

Khi mô hình đang chạy trên Server, kết quả sẽ liên tục được lữu tại thư mục `results/`. Để xem hoặc lưu trữ kết quả này từ xa, bạn có thể dùng 1 trong 3 cách sau:

### Cách 1: Kéo trực tiếp thư mục results về máy tính (Đơn giản nhất)
Mở một Tab Terminal khác trên **máy cá nhân (Local)** của bạn và chạy lệnh rsync ngược lại từ server về:

```bash
# Nhớ đổi IP và Port cho đúng với Server
rsync -avh --progress -e "ssh -p 42341" \
  root@203.0.113.10:/workspace/DSP-CS-ASR/recipes/DSP_CodeSwitch/ASR/results/ \
  /home/hnn/Documents/kltn/DSP-CS-ASR/recipes/DSP_CodeSwitch/ASR/results_from_server/
```
Lệnh này chỉ tải về những file mới hoặc có thay đổi (rất nhanh). Bạn có thể chạy lệnh này bất cứ lúc nào để cập nhật kết quả mới nhất.

### Cách 2: Dùng VS Code Remote SSH + TensorBoard (Xem biểu đồ Real-time)
Nếu bạn cấu hình VS Code để kết nối Remote SSH thẳng vào Vast AI:
1. Mở Terminal của VS Code (trên server), chạy lệnh:
   ```bash
   tensorboard --logdir=./recipes/DSP_CodeSwitch/ASR/results
   ```
2. VS Code sẽ tự động nhận diện ứng dụng Web chạy ở cổng 6006 và hỏi bạn có muốn mở trên trình duyệt (thông qua tính năng Port Forwarding tích hợp). Ở máy nội bộ, bạn chỉ cần mở `http://localhost:6006` để xem đồ thị real-time.

### Cách 3: Đẩy trực tiếp lên Google Drive bằng rclone
Nếu bạn sợ server sập mất kết quả, bạn có thể tự động sao lưu lên Google Drive.
Trên Server, cài đặt `rclone`:
```bash
sudo apt install rclone -y
rclone config  # Làm theo hướng dẫn trên màn hình để liên kết với tài khoản Google Drive
```
Cấu hình xong, bạn có thể bật Auto-sync (ví dụ mỗi giờ đồng bộ 1 lần):
```bash
# Đẩy thư mục kết quả vào thư mục 'KLTN_Run_v1' trên Google Drive mỗi 3600s
watch -n 3600 "rclone sync ./recipes/DSP_CodeSwitch/ASR/results/ Mydrive:KLTN_Run_v1/"
```
