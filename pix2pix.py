import os # Dosya ve klasör işlemleri için (veri yükleme vs.)
import re  # 👈 Dosya isimlerinden sayı çekmek için eklendi
import torch  #PyTorch ana kütüphane - tensörler ve model eğitimi için
import torch.nn as nn  # Derin öğrenme katmanlarını oluşturmak için (CNN, ReLU vb.)
from torchvision import transforms  # Görsel dönüşüm işlemleri için (resize, normalize vb.)
from torch.utils.data import Dataset, DataLoader # Verileri batch'leyip modele vermek için
from PIL import Image # Verileri batch'leyip modele vermek için

# -------------------------
# Generator
# -------------------------
class Generator(nn.Module):  #nn.Module, tüm modellerin türediği ana sınıftır. generator bu sınfı miras alır
    def __init__(self, latent_dim=3): #rgb görüntüler için 3 kanal 
        super(Generator, self).__init__()
        #Tüm katmanları sıralı bir şekilde tanımlarız.- nn.Sequential, bir dizi katmanı sırayla uygular (ileri besleme yönünde) 
        self.model = nn.Sequential(
            #konvülasyon katmanı
            #2D görüntüler için konvolüsyon (filtreleme) işlemi yapar
            #kernel_size=4: Filtre boyutu 4x4
            #İki pikselde bir kayarak görüntüyü yarı boyuta indirir.
            nn.Conv2d(latent_dim, 64, 4, stride=2, padding=1),

            #relu aktivasyon fonksiyonudur ve negatif degerler 0 yapar, true ise belleği optimize eder
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),

            #Batch Normalization, öğrenmeyi hızlandırır ve denge sağlar.128 kanalı normalize eder
            nn.BatchNorm2d(128),

            nn.ReLU(True),

            #Konvolüsyonun Tersi (TransposeConv)
            #bu katman görüntüyü yeniden büyütür 
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),

            #egitim sırasında veriler normalize edilmistir. bu yüzden çıktıyı Çıktıyı [-1, 1] aralığına sıkıştırır.
            nn.Tanh()
        )

    #Bu fonksiyon modelin çalışmasını tanımlar.
    def forward(self, x):
        return self.model(x)

# -------------------------
# Discriminator (PatchGAN)
# -------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        #Katman Oluşturucu Fonksiyon (block):Bu fonksiyon, sık tekrar edilen katmanları oluşturmak için tanımlanmış mini bir fonksiyondur.
        #LeakyReLU: Aktivasyon fonksiyonu. Negatif değerleri sıfırlamak yerine küçük bir oranda geçirir (0.2).
        def block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers #Bu fonksiyon bir liste döndürür: Bu katmanlar tek başına kullanılmaz. Genellikle nn.Sequential(*block(...)) şeklinde birleştirilerek bir modelin parçası yapılır.

        self.model = nn.Sequential(
            #Girdi olarak hem orijinal resim hem de çıktı resim (gerçek ya da sahte) veriyoruz.bu yüzden çarpı 2
             #Yani: 3 kanal (girdi) + 3 kanal (çıktı) = 6 kanal
             #İlk katman: Normalize edilmez (genelde ilk katmanda yapılmaz).
             #Girdi: 6 kanal → Çıktı: 64 kanal
             #Görsel boyutu küçülür.
            *block(in_channels * 2, 64, normalize=False),

            #64 → 128 kanala çıkar. Normalize edilir.
            *block(64, 128),

            #128 → 256 kanal
            *block(128, 256),
            *block(256, 512),

             #512 kanal → 1 kanallı sonuç haritası üretir.
            #Bu çıktı, her bölgenin "gerçek mi sahte mi" olduğuna dair tahminidir.
            #Bu yaklaşıma PatchGAN denir: Görselin tamamı yerine küçük parçalar değerlendirilir.
            nn.Conv2d(512, 1, 4, padding=1)
        )



    #img_A: Girdi görseli
    #img_B: Çizim görseli (gerçek ya da üretilmiş)
    #torch.cat((img_A, img_B), 1): Bu iki resmi kanal boyutunda birleştirir (örneğin 3+3 = 6 kanal olur)
    #self.model(x): Bu birleşik resmi modele verir ve sonucu döndürür.
    def forward(self, img_A, img_B):
        x = torch.cat((img_A, img_B), 1)
        return self.model(x)





# -------------------------
# Ayarlar
# -------------------------
epochs = 10
batch_size = 4
img_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Veri Dönüştürme ve Yükleme
# -------------------------
#Bu, görseli modele vermeden önce nasıl işleyeceğimizi tanımlar.
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  #reesize, Görseli 256x256 boyutuna getirir. çünkü yukarda öyle ayarlamıstık
    transforms.ToTensor() #tetensor PIL formatındaki görseli PyTorch’un anlayacağı tensör formatına çevirir.Ayrıca pikselleri [0, 1] arasına normalize eder.
])

# Dosya isimlerini içindeki sayıya göre sıralamak için
def sort_by_number(file_list):
    return sorted(file_list, key=lambda x: int(re.findall(r'\d+', x)[0]))

class PairedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.normal_dir = os.path.join(root_dir, "girdi")
        self.sketch_dir = os.path.join(root_dir, "cikti")
        
        self.normal_images = sort_by_number(os.listdir(self.normal_dir))  
        self.sketch_images = sort_by_number(os.listdir(self.sketch_dir))  
        
        #Eğer bir klasörde fazla görüntü varsa, eşleşmenin bozulmaması için daha az olan kadar veri kullanılır.
        self.length = min(len(self.normal_images), len(self.sketch_images))
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx): #TAM YOL VERİR
        normal_path = os.path.join(self.normal_dir, self.normal_images[idx])
        sketch_path = os.path.join(self.sketch_dir, self.sketch_images[idx])

        normal_image = Image.open(normal_path).convert("RGB")
        sketch_image = Image.open(sketch_path).convert("RGB")

        if self.transform:
            normal_image = self.transform(normal_image)
            sketch_image = self.transform(sketch_image)

        return normal_image, sketch_image #Sonuç olarak bir (girdi, çıktı) çifti döner.


# Dosya yolları buraya göre düzenlenmiş:
dataset = PairedDataset(r"C:\Users\gulse\Desktop\karakalem", transform=transform)

#shuffle=True: Her epoch başında verileri karıştırır. Böylece model ezberlemeye meyilli olmaz.
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------
# Model, optimizer ve loss
# -------------------------
#Daha önce tanımlanan Generator ve Discriminator sınıflarından birer model nesnesi oluşturuluyor.
generator = Generator().to(device)
discriminator = Discriminator().to(device)

#İki adet optimizer (ağırlıkları güncelleyen algoritma) tanımlanıyor
#Biri Generator için (optimizer_G), diğeri Discriminator için (optimizer_D).
#Adam optimizasyon algoritması kullanılıyor.
#lr=0.0002: Öğrenme oranı — ağırlıkların ne kadar değişeceğini belirler.
#betas=(0.5, 0.999): Adam algoritmasının moment hesaplamaları için sabitler (genellikle bu değerler önerilir GAN eğitimlerinde).
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#Bu iki kayıp fonksiyonu birlikte Generator'ı eğitmekte kullanılır:

#1) adversarial_loss: Generator'ın Discriminator'ı kandırmak için ürettiği görsellerin başarısını ölçer (Mean Squared Error).
#Düşük olması iyi generator iyi gorsel uretiyor demektir 
adversarial_loss = nn.MSELoss()


#2) pixelwise_loss: Gerçek ve üretilen görseller arasında piksel seviyesinde farkı ölçer (Mean Absolute Error / L1).
pixelwise_loss = nn.L1Loss()

# -------------------------
# Eğitim Döngüsü
# -------------------------
for epoch in range(epochs):
    for i, batch in enumerate(dataloader):
        if batch is None:
            continue
        real_A, real_B = batch
        #Görselleri modelin çalıştığı cihaza (CPU veya GPU) taşıyoruz.
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        #valid: Gerçek görsel için etiket. Tüm değerleri 1.
        #fake: Sahte (üretilmiş) görsel için etiket. Tüm değerleri 0.
        #Bu etiketler Discriminator'ın çıktılarına karşılık gelir.
        #1-> 1 SKOR DOĞRI YA DA YANLİS
        #PatchGAN mimarisine göre Discriminator çıktısının boyutu
        #PatchGAN Görüntüyü 15x15’lik bir ızgara (grid) gibi düşünür.HER BİR İÇİN 1 YA SDA 0 SKORU OLUSTURUR
        valid = torch.ones((real_A.size(0), 1, 15, 15), requires_grad=False).to(device)
        fake = torch.zeros((real_A.size(0), 1, 15, 15), requires_grad=False).to(device)

        # Train Generator
        #Generator'ın önceki eğitim adımlarından gelen gradyanları sıfırlıyoruz
        #Bu, her eğitim adımında doğru gradyanların hesaplanmasını sağlar.
        optimizer_G.zero_grad()

        fake_B = generator(real_A) #Generator, real_A (girdi görüntüsü) ile fake_B (sahte çıktı görseli) üretir.
        g_loss = adversarial_loss(discriminator(real_A, fake_B), valid) + 100 * pixelwise_loss(fake_B, real_B)
        
        #Generator kaybının gradyanları hesaplanır. Bu işlem, modelin ağırlıklarını güncellemek için gereklidir.
        g_loss.backward()

        #Generator'ın ağırlıkları, gradyanları kullanarak güncellenir.
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        #gerçek kaybı (real_loss): Gerçek görseller (real_A, real_B) Discriminator tarafından test edilir ve valid etiketiyle karşılaştırılır.
        #Sahte kaybı (fake_loss): Generator tarafından üretilen sahte görseller (real_A, fake_B.detach()) Discriminator tarafından test edilir ve fake etiketiyle karşılaştırılır.
        #.detach() kullanılması, sahte görsellerin gradyanlarının Generator’a geri gitmesini engeller.
        real_loss = adversarial_loss(discriminator(real_A, real_B), valid)
        fake_loss = adversarial_loss(discriminator(real_A, fake_B.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

# Modeli kaydet
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
print("✅ Eğitim tamamlandı, modeller kaydedildi.")
