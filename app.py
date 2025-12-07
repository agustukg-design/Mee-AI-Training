import os
import torch
import torchaudio

print("=== MEMULAI SISTEM MEE AI ARTIFICIAL ===")

# 1. Cek Mesin GPU (Otak AI)
if torch.cuda.is_available():
    print(f"✅ GPU DETEKSI: {torch.cuda.get_device_name(0)}")
    print("   Status: SIAP UNTUK TRAINING")
else:
    print("❌ BAHAYA: GPU TIDAK TERDETEKSI!")

# 2. Cek Data Audio (Bahan Bakar)
audio_folder = "FOLDER INDUK AUDIO ALKITAB BHS MEE"
if os.path.exists(audio_folder):
    files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3') or f.endswith('.wav')]
    print(f"✅ DATA AUDIO DITEMUKAN: {len(files)} file audio siap diproses.")
else:
    print(f"❌ DATA AUDIO TIDAK DITEMUKAN: Folder '{audio_folder}' tidak ada.")

print("=== SISTEM SIAP. MENUNGGU PERINTAH LANJUTAN ===")