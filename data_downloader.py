import kagglehub, shutil, os

# Tải dataset
path = kagglehub.dataset_download("nguynrichard/auto-vqabest")

# Thư mục đích
target = "/root/modeltuner/modeltuner/data"

# Tạo thư mục nếu chưa có
os.makedirs(target, exist_ok=True)

# Copy toàn bộ nội dung
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Dataset đã copy vào:", target)
