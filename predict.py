import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def load_and_preprocess_clinical_data(clinical_data_path):
    df = pd.read_csv(clinical_data_path)
    features = df.drop(columns=['Filename']).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features

def main():
    image_path = 'path_to_new_image.jpg'
    clinical_data_path = 'path_to_new_clinical_data.csv'

    image = load_and_preprocess_image(image_path)
    clinical_data = load_and_preprocess_clinical_data(clinical_data_path)

    model = load_model('alzheimers_model.h5')
    prediction = model.predict([image, clinical_data])
    print(f'Prediction: {prediction[0][0]:.4f}')

if __name__ == '__main__':
    main()
