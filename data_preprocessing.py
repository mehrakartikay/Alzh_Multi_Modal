import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_images(image_dir, image_size=(224, 224)):
    images = []
    filenames = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(image_size)
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            filenames.append(filename)
    return np.array(images), filenames

def load_clinical_data(csv_path, filenames):
    df = pd.read_csv(csv_path)
    df['Filename'] = df['Filename'].astype(str)
    df = df[df['Filename'].isin(filenames)]
    df = df.set_index('Filename').loc[filenames]
    features = df.drop(columns=['Label']).values
    labels = df['Label'].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, labels

def main():
    image_dir = 'mri_images'
    csv_path = 'clinical_data.csv'

    images, filenames = load_images(image_dir)
    clinical_features, labels = load_clinical_data(csv_path, filenames)

    np.save('images.npy', images)
    np.save('clinical_features.npy', clinical_features)
    np.save('labels.npy', labels)

if __name__ == '__main__':
    main()
