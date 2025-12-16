import os

def rename_bins():
    folder_path = 'bins'
    extensions = ('.jpg', '.jpeg', '.png')
    
    try:
        # 1. Aşama: Dosyaları listele
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
        print(f"{len(files)} adet resim bulundu. Çakışmaları önlemek için işlem yapılıyor...")

        # 2. Aşama: Önce hepsine geçici benzersiz isimler ver (Çakışmayı önlemek için)
        temp_files = []
        for index, filename in enumerate(files):
            ext = os.path.splitext(filename)[1]
            old_path = os.path.join(folder_path, filename)
            temp_name = f"temp_rename_{index}{ext}"
            temp_path = os.path.join(folder_path, temp_name)
            os.rename(old_path, temp_path)
            temp_files.append(temp_name)

        # 3. Aşama: Şimdi hepsini 1, 2, 3 diye sırala
        for index, filename in enumerate(temp_files, start=1):
            ext = os.path.splitext(filename)[1]
            old_path = os.path.join(folder_path, filename)
            new_name = f"{index}{ext}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Başarılı: -> {new_name}")

        print("\nTamamlandı! Tüm dosyalar 1'den başlayarak sıralandı.")
        
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    rename_bins()