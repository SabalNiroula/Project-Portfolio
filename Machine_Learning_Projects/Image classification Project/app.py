import streamlit as st
from PIL import Image
import tensorflow as tf
import pickle
import io
import json

le = pickle.load(open('label_encoder.pkl', 'rb'))
model = tf.keras.models.load_model('model.h5')

# read the json file
with open('remedies.json', 'r') as f:
    data = json.load(f)

st.set_page_config(page_title="Plant Disease Prediction",
                   page_icon=":herb:", layout="wide")

def predict_disease(img):
  # convert PIL type into bytes type
  img_bytes = io.BytesIO()
  img.save(img_bytes, format='jpeg')
  img_bytes = img_bytes.getvalue()

  # Decode the image
  img = tf.image.decode_jpeg(img_bytes, channels=3)

  img = tf.image.resize(img, (64, 64))
  img = tf.reshape(img, (1, 64, 64, 3))
  img = img/255.0
  prediction = model.predict(img)
  predicted_labels = le.inverse_transform(prediction.argmax(axis=1))
  confidence = tf.reduce_max(prediction)
  return predicted_labels[0], confidence


def format_disease(label):
    predicted_disease = label.split("___")
    name = predicted_disease[0]
    disease = " ".join(predicted_disease[1].split("_"))
    return name, disease


def get_remedy(label):
    if label in data:
        return data[label]
    else:
        return "No remedies found for this disease"


st.title("Plant Disease Prediction App")

uploaded_file = st.file_uploader(
    "Upload an image of a plant", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.")

    if st.button("Predict"):
        # make prediction
        prediction = predict_disease(image)
        name, disease = format_disease(prediction[0])
        if disease == 'healthy':
            st.success("The plant is healthy \nConfidence : {}".format(
                round(prediction[1].numpy()*100, 2)))

        else:
            st.success(
                "Disease of {} and the disease name is {} \n Confidence : {}".format(name, disease, round(prediction[1].numpy()*100, 2)))
            st.write(get_remedy(prediction[0]))