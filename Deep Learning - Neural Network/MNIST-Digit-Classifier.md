# MNIST Digit Classifier with Streamlit

This is a beginner-friendly **Deep Learning + Streamlit mini project** that allows you to draw digits (0–9) on a canvas and lets a **Convolutional Neural Network (CNN)** predict them in real time.


---

## Concepts Covered
1. **Data Science workflow** → dataset, preprocessing, training, evaluation  
2. **Convolutional Neural Networks (CNNs)** → Conv2D, MaxPooling, Flatten, Dense  
3. **Training MNIST dataset** → Handwritten digits recognition  
4. **Streamlit basics** → interactive web app for ML models  
5. **Drawable Canvas** → drawing digits and feeding into the model  

---

## Installation & Setup

### Step 1: Create a virtual environment (optional but recommended)
Let's create a folder named `Deep Learning` and Create a virtual environment as follows:
```bash
cd Deep Learning
python -m venv venv
.\venv\Scripts\activate
```

### Step 2: Install the dependencies

Create a requirements.txt inside the `Deep Learning` project root folder with the following:

```python
streamlit
tensorflow
numpy
pillow
streamlit-drawable-canvas
```
### Now install:

```bash
pip install -r requirements.txt
```

### Step 3: Understanding MNIST
- MNIST: 70k grayscale images of handwritten digits (28×28). Values are in 0–255; we normalize to 0–1.

- CNN (Convolutional Neural Network):

    - Conv2D: Learns filters to detect strokes/edges.
    - MaxPooling2D: Downsamples (keeps salient features).
    - Flatten: Converts 2D feature maps into a vector.
    - Dense: Fully connected layers for classification.
    - softmax: Outputs probabilities across the 10 classes.

Mini example (concept):
- If your input pixel is 255 (white), dividing by 255 → 1.0. Black background is 0.0.
- The first conv layer might learn a vertical-edge detector that fires strongly on vertical strokes of “1”, “4”, etc.

## Project Implementation
Create a main project file name app.py and place the below codes one by one:
###  Step 4: Import Libraries

```python
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from streamlit_drawable_canvas import st_canvas
from PIL import Image
```

### Step 5: Load/Train Model

```python
@st.cache_resource
def load_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=128, verbose=1)
    return model

model = load_model()
```

### Step 6: Create Streamlit UI

```python
st.title("🖌️ MNIST Digit Classifier")
st.write("Draw a digit (0–9) in the canvas below and let the model predict it!")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)
```

### Step 7: Make Prediction

```python
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28)).convert("L")

        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.subheader(f"🎯 Predicted Digit: **{predicted_class}**")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit before predicting.")

```

## Run the app
```bash
streamlit run app.py
```