import streamlit as st  # Streamlit kütüphanesi, web arayüzü oluşturmak için kullanılır
import openai  # OpenAI API'sini kullanarak DALL·E gibi modellerle iletişim kurmak için kullanılır
from PIL import Image  # Pillow kütüphanesi, görsellerle çalışmak için
import torch  # PyTorch kütüphanesi, modelin çalışması için gerekli
from torchvision import transforms  # Görsel dönüştürme işlemleri için
import requests  # HTTP istekleri göndermek için (görseli DALL·E'den çekmek için)
from io import BytesIO  # Bayt formatındaki verileri dosya gibi işlemek için
from pix2pix import Generator  # Pix2Pix GAN modeli içinde tanımlı Generator sınıfı

# --- OpenAI API Anahtarı ---
openai.api_key = ""  # OpenAI API erişimi için gerekli anahtar (güvenliğiniz için gizli tutun)

# --- PyTorch Cihaz Ayarı ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# Eğer GPU varsa 'cuda', yoksa 'cpu' cihazı kullanılır. Model buna göre yüklenir.

# --- Generator Modelini Yükle ---
generator = Generator().to(device)  # Generator modeli oluşturulup ilgili cihaza gönderilir
generator.load_state_dict(torch.load("generator.pth", map_location=device))  
# Eğitilmiş model ağırlıkları 'generator.pth' dosyasından yüklenir
generator.eval()  # Model 'evaluation' moduna alınır (dropout, batchnorm vs. devre dışı)

# --- Görseli Karakaleme Çeviren Fonksiyon ---
def generate_sketch(image):  # Bu fonksiyon girilen görseli karakalem çizime dönüştürür
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Görseli 256x256 boyutuna getirir
        transforms.ToTensor()  # Görseli [0,1] aralığında tensöre çevirir
    ])
    input_image = transform(image).unsqueeze(0).to(device)  
    # Görselin boyutları [1, 3, 256, 256] olacak şekilde düzenlenip cihaza aktarılır

    with torch.no_grad():  # Bu blokta grad hesaplaması yapılmaz, bu da hız ve bellek kazancı sağlar
        output = generator(input_image)  # Görsel Generator modeline verilir ve çıktı alınır

    output_image = output.squeeze().cpu().clamp(0, 1)  
    # Çıktı tensörü CPU'ya alınır, boyutu sıkıştırılır, değerler 0-1 aralığına kısıtlanır
    return transforms.ToPILImage()(output_image)  # Tensor tekrar PIL görseline çevrilip döndürülür

# --- Streamlit Arayüzü ---
st.title("Yapay Zeka ile Görsel Üret + Karakalem Çizim 🎨")  
# Sayfa başlığı belirlenir

# --- Prompt Girişi ---
prompt = st.text_input("Ne görseli istersiniz ")  
# Kullanıcıdan metin girişi alınır (DALL·E için prompt)

# --- Görsel Üret Butonu ---
if st.button("Görsel Üret"):  # Bu butona tıklandığında aşağıdaki işlemler gerçekleşir
    if prompt:  # Eğer kullanıcı boş prompt vermediyse
        with st.spinner("DALL·E görsel oluşturuyor..."):  # İşlem süresince spinner gösterilir
            response = openai.Image.create(
                prompt=prompt,  # Kullanıcının yazdığı metin (prompt) modele verilir
                n=1,  # Kaç görsel oluşturulacağı (1 tane)
                size="512x512"  # Görsel boyutu (512x512 önerilen boyuttur)
            )
            image_url = response["data"][0]["url"]  # Oluşturulan görselin URL'si çekilir
            image_response = requests.get(image_url)  # URL üzerinden görsel indirilir
            dalle_image = Image.open(BytesIO(image_response.content)).convert("RGB")  
            # Görsel RGB formatında PIL Image olarak açılır
            st.image(dalle_image, caption="DALL·E Tarafından Oluşturulan Görsel", use_container_width=True)  
            # Görsel arayüzde gösterilir (yeni parametre: use_container_width)

            # Görseli oturumda sakla
            st.session_state["generated_image"] = dalle_image  # Görseli session'da saklıyoruz
    else:
        st.error("Lütfen bir prompt girin.")  # Eğer prompt boşsa kullanıcı uyarılır

# --- Karakalem Butonu ---
if "generated_image" in st.session_state:  # Eğer önceki aşamada bir görsel oluşturulduysa
    if st.button("Karakalem Çizime Dönüştür"):  # Bu butona basıldığında aşağıdaki işlemler yapılır
        with st.spinner("Karakalem çizim hazırlanıyor..."):  # İşlem süresince spinner gösterilir
            sketch = generate_sketch(st.session_state["generated_image"])  # Görseli karakalem haline çevir
            st.image(sketch, caption="Karakalem Çizim", use_container_width=True)  # Sonuç arayüzde gösterilir
            sketch.save("generated_sketch_from_prompt.jpg")  # Görsel diske kaydedilir
            st.success("Karakalem çizim tamamlandı!")  # Kullanıcıya başarı mesajı gösterilir
