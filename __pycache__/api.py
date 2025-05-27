import requests

url = "https://ai-text-to-image-generator-flux-free-api.p.rapidapi.com/aaaaaaaaaaaaaaaaaiimagegenerator/fluximagegenerate/generateimage.php"

payload = {
    "prompt": "Cyberpunk city at night",
    "width": "512",
    "height": "512",
    "seed": "123456",
    "model": "flux"
}

headers = {
    "x-rapidapi-key": "74ece91c4cmshc344f4765cd961ep1167b0jsnb3bc4a3069ca",
    "x-rapidapi-host": "ai-text-to-image-generator-flux-free-api.p.rapidapi.com",
    "Content-Type": "application/x-www-form-urlencoded"
}

response = requests.post(url, data=payload, headers=headers)

print("Durum Kodu:", response.status_code)
print("Cevap:", response.text)
