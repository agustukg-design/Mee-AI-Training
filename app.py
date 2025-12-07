import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import time

# --- KONFIGURASI MEE AI ---
DATA_PATH = "FOLDER INDUK AUDIO ALKITAB BHS MEE"
EPOCHS = 500        # Putaran belajar
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 MEMULAI TRAINING MEE AI (MEE-CHAT) MENGGUNAKAN: {torch.cuda.get_device_name(0)}")

# 1. PERSIAPAN DATA (Audio Reader)
class MeeAudioDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mp3') or f.endswith('.wav')]
        # Mengubah suara menjadi gambar matematika (Mel Spectrogram)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_mels=64).to(DEVICE)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            waveform, sample_rate = torchaudio.load(self.files[idx])
            # Paksa stereo jadi mono agar hemat memori
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Potong audio panjang menjadi potongan kecil (3 detik pertama) untuk tes
            # Agar memori GPU tidak meledak
            max_len = 48000 * 3 
            if waveform.shape[1] > max_len:
                waveform = waveform[:, :max_len]
            elif waveform.shape[1] < max_len:
                # Padding jika terlalu pendek (isi dengan nol/hening)
                pad_size = max_len - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))

            waveform = waveform.to(DEVICE)
            spec = self.mel_transform(waveform)
            return spec.transpose(1, 2) # (Time, Freq)
        except Exception as e:
            print(f"⚠️ File rusak dilewati: {self.files[idx]}")
            return torch.zeros(1, 100, 64).to(DEVICE) # Return dummy

# 2. DEFINISI OTAK AI (Neural Network - LSTM)
class MeeBrain(nn.Module):
    def __init__(self):
        super(MeeBrain, self).__init__()
        # LSTM layer untuk mempelajari urutan suara
        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=2, batch_first=True)
        # Output layer untuk merekonstruksi suara
        self.fc = nn.Linear(256, 64) 
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# 3. EKSEKUSI TRAINING (Jantung Program)
def train():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Folder {DATA_PATH} tidak ditemukan!")
        return

    dataset = MeeAudioDataset(DATA_PATH)
    print(f"📚 Total Data Audio: {len(dataset)} file. Sedang memuat ke dalam VRAM...")
    
    if len(dataset) == 0:
        print("❌ Tidak ada file audio .mp3 atau .wav ditemukan.")
        return

    model = MeeBrain().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("🔥 TRAINING DIMULAI! Perhatikan 'Loss' (Angka Error) yang harus MENURUN.")
    print("-" * 50)
    
    model.train()
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        total_loss = 0
        count = 0
        
        # Ambil sampel (Batch processing simulasi)
        for i in range(min(len(dataset), 20)): # Latih 20 file pertama dulu
            spec = dataset[i].squeeze(0) # (Time, Freq)
            input_data = spec.unsqueeze(0) # (Batch, Time, Freq)
            
            optimizer.zero_grad()
            output = model(input_data)
            
            # Kita latih AI untuk mereproduksi suara input (Autoencoder task)
            loss = criterion(output, input_data) 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        end_time = time.time()
        
        print(f"Epoch [{epoch}/{EPOCHS}] | 📉 Error (Loss): {avg_loss:.6f} | ⏱️ Waktu: {end_time - start_time:.2f} detik")
        
        if epoch % 50 == 0:
            torch.save(model.state_dict(), "mee_ai_model.pth")
            print(f"💾 Model Backup Disimpan: mee_ai_model.pth")

    print("-" * 50)
    print("✅ TRAINING SELESAI! Model dasar telah terbentuk.")

if __name__ == "__main__":
    train()