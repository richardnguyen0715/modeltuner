import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import zipfile

# --- Thư mục local cần upload ---
local_folder = "/root/auto_vivqa/modeltuner/checkpoints"
zip_filename = "/root/auto_vivqa/modeltuner/checkpoints.zip"

# --- Nếu muốn upload vào 1 folder cụ thể trên Drive ---
# Tạo folder trước trên Google Drive, lấy ID từ URL:
# Ví dụ: https://drive.google.com/drive/folders/<FOLDER_ID>
folder_id = "1cLo7Jsu6Z94GBdGHZx6Z_jrcxOZqoyzP"  # thay bằng ID folder thật trên Drive

# Tạo zip
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            filepath = os.path.join(root, file)
            # Đường dẫn lưu trong zip (relative path)
            arcname = os.path.relpath(filepath, local_folder)
            zipf.write(filepath, arcname)
print(f"✅ Folder zipped to {zip_filename}")

# --- Xác thực với Google ---
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # mở link -> copy mã -> dán vào terminal
drive = GoogleDrive(gauth)

print("Xác thực thành công với Google Drive.")

# --- Upload tất cả file ---
gfile = drive.CreateFile({'title': os.path.basename(zip_filename),
                          'parents': [{'id': folder_id}]})
gfile.SetContentFile(zip_filename)
gfile.Upload()
print(f"✅ Uploaded {zip_filename} to Google Drive folder {folder_id}")