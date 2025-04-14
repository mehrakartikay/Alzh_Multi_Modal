from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model

def create_model(clinical_features_shape):
    # Image Branch
    image_input = Input(shape=(224, 224, 3), name='image_input')
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Clinical Data Branch
    tabular_input = Input(shape=(clinical_features_shape,), name='tabular_input')
    y = Dense(64, activation='relu')(tabular_input)
    y = Dropout(0.3)(y)

    # Combine Branches
    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.4)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[image_input, tabular_input], outputs=output)
    return model
