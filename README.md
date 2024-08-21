# CDIP-ChatGLM3
## Crop disease identification and prescription using CNNs, ChatGLM3-6b, and Fine-tuned models

https://github.com/user-attachments/assets/a098d6a3-00a9-4350-abe1-4187ef419887


https://github.com/user-attachments/assets/17621a5a-9d07-49d8-9760-ac859c511b9e


https://github.com/user-attachments/assets/e2cac48e-b38c-42f7-a41e-f4dcf88bf9b4

## Dataset
### Supported crop types include:
 Apple, Cherry, Citrus, Corn, Grape, Orange, Peach, Pepper, Potato, Rice, Soybean, Strawberry, Tomato, Wheat.
### Supported crop disease types include: 
Apple Alternaria Boltch, Apple Black rot, Apple Brown Spot, Apple Grey spot, Apple healthy, Apple Mosaic, Apple Powdery, Apple Rust, Cherry healthy, Cherry Powdery Mildew general, Citrus Greening June general, Citrus Greening June serious, Citrus healthy, Corn Common rust, Corn Gray leaf spot, Corn Northern Leaf Blight, Grape Leaf Blight, Grape mosaic virus, Grapevine yellow, Grape Black rot, Grape downy mildew, Grape healthy, Grape Powdery, Grape Esca (Black Measles), Orange Haunglongbing (Citrus greening), Peach Bacterial spot, Peach healthy, Pepper healthy, Pepper scab general, Pepper bell Bacterial spot, Pepper bell healthy, Potato healthy, Potato Blight Fungus, Rice Bacterial blight, Rice Brown Spot, Rice Healthy, Rice Hispa, Rice Leaf Blast, Rice Tungro, Soybean healthy, Strawberry healthy, Strawberry Leaf scorch, Tomato Bacterial spot, Tomato Early blight, Tomato healthy, Tomato Late blight, Tomato Leaf Mold, Tomato mosaic virus, Tomato Septoria leaf spot, Tomato Yellow Leaf Curl Virus, Wheal Leaf Rust, Wheat Healthy.

## Model training
### Development environment setup
```python
conda create -n chatglm python=3.8
conda activate chatglm
cd AgriAgent
pip install -r requirements.txt
