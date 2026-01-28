import requests

url = "https://r2-public-models.ultralytics.com/car-damage-yolov8s.pt"
output_path = "car_damage_s.pt"

print("Downloading model...")

response = requests.get(url, stream=True)
response.raise_for_status()

with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("Download complete →", output_path)
