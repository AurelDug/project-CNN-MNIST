import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset

ds = load_dataset(
    "ChristophSchuhmann/MS_COCO_2017_URL_TEXT",
    split="train",
    streaming=True
)

for row in ds:
    url = row["URL"]
    txt = row["TEXT"]

    response = requests.get(url)
    #img = Image.open(BytesIO(response.content))

    #img.show()
    print(txt)

