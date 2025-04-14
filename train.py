import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from plot_results import plot_training_results  # Import your plotting function

def main():
    # Load data
    images = np.load('images.npy')
    clinical_features = np.load('clinical_features.npy')
    labels = np.load('labels.npy')

    # Split the data into training and validation sets
    X_img_train, X_img_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
        images, clinical_features, labels, test_size=0.2, random_state=42)

    # Create and compile the model
    model = create_model(clinical_features_shape=clinical_features.shape[1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks for early stopping and reducing learning rate on plateau
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3)
    ]

    # Train the model and capture the training history
    history = model.fit(
        [X_img_train, X_tab_train], y_train,
        validation_data=([X_img_val, X_tab_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )

    # Plot the training results (loss and accuracy)
    plot_training_results(history)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate([X_img_val, X_tab_val], y_val)
    print(f'\nFinal Validation Accuracy: {val_accuracy:.4f}')

    # Save the trained model
    model.save('alzheimers_model.keras')

if __name__ == '__main__':
    main()
