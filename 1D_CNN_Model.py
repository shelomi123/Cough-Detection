#1
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from google.colab import drive
import zipfile
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

SAVE_DIR = "/content/extracted_data/preprocessed_data/"
OUTPUT_DIR = "/content/drive/MyDrive/ML_Project_bestmodel/"  # Updated output directory


# Mount your Google Drive
drive.mount('/content/drive')

# Path to your zip file in Google Drive
zip_file_path = '/content/drive/MyDrive/ML_Project_bestmodel/processed_output.zip'

# Create a directory to extract the files
extract_dir = '/content/extracted_data'
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    print("Files extracted successfully!")


# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Output directory created/verified: {OUTPUT_DIR}")

class CoughCNN:
    def __init__(self):
        """Initialize the 1D CNN model for cough classification"""
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None

    def load_preprocessed_data(self, data_path):
        """
        Load preprocessed audio data from your folder structure

        Args:
            data_path (str): Path to processed_output folder

        Returns:
            dict: Dictionary with train, val, test splits
        """
        data_splits = {}

        for split in ['train', 'val', 'test']:
            X_split = []
            y_split = []

            split_path = os.path.join(data_path, split)

            # Load cough samples
            cough_path = os.path.join(split_path, 'cough')
            if os.path.exists(cough_path):
                cough_files = [f for f in os.listdir(cough_path) if f.endswith('.wav')]
                print(f"Found {len(cough_files)} cough files in {split}")

                for file in cough_files:
                    # Load preprocessed audio data
                    audio_data = self.load_audio_array(os.path.join(cough_path, file))
                    if audio_data is not None:
                        X_split.append(audio_data)
                        y_split.append('cough')

            # Load non_cough samples
            non_cough_path = os.path.join(split_path, 'non_cough')
            if os.path.exists(non_cough_path):
                non_cough_files = [f for f in os.listdir(non_cough_path) if f.endswith('.wav')]
                print(f"Found {len(non_cough_files)} non_cough files in {split}")

                for file in non_cough_files:
                    # Load preprocessed audio data
                    audio_data = self.load_audio_array(os.path.join(non_cough_path, file))
                    if audio_data is not None:
                        X_split.append(audio_data)
                        y_split.append('non_cough')

            if X_split:
                data_splits[split] = {
                    'X': np.array(X_split),
                    'y': np.array(y_split)
                }
                print(f"{split} set: {len(X_split)} samples")

        return data_splits

    def load_audio_array(self, file_path):
        """
        Load preprocessed audio data

        Args:
            file_path (str): Path to audio file

        Returns:
            np.array: Preprocessed audio array
        """
        try:

            import librosa
            audio, _ = librosa.load(file_path, sr=None)  # Load with original sample rate
            return audio

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def prepare_data(self, data_splits):
        """
        Prepare data for training

        Args:
            data_splits (dict): Data splits dictionary

        Returns:
            tuple: Prepared train, val, test data
        """
        # Fit label encoder on all labels
        all_labels = []
        for split_data in data_splits.values():
            all_labels.extend(split_data['y'])

        self.label_encoder.fit(all_labels)
        print(f"Classes: {self.label_encoder.classes_}")

        prepared_data = {}

        for split, split_data in data_splits.items():
            X = split_data['X']
            y = split_data['y']

            # Reshape for 1D CNN if needed (samples, timesteps, features)
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], X.shape[1], 1)

            # Encode labels to categorical
            y_encoded = self.label_encoder.transform(y)
            y_categorical = to_categorical(y_encoded, num_classes=len(self.label_encoder.classes_))

            prepared_data[split] = {
                'X': X,
                'y': y_categorical
            }

            print(f"{split} - X shape: {X.shape}, y shape: {y_categorical.shape}")

        return prepared_data

    def build_model(self, input_shape, num_classes=2):
        """
        Build 1D CNN architecture

        Args:
            input_shape (tuple): Shape of input (timesteps, features)
            num_classes (int): Number of classes
        """
        model = models.Sequential([
            # First Conv Block
            layers.Conv1D(32, kernel_size=15, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(32, kernel_size=8, activation='relu'),
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.25),

            # Second Conv Block
            layers.Conv1D(64, kernel_size=11, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(64, kernel_size=6, activation='relu'),
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.25),

            # Third Conv Block
            layers.Conv1D(128, kernel_size=7, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=4, activation='relu'),
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.25),

            # Fourth Conv Block
            layers.Conv1D(256, kernel_size=5, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.5),

            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),

            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])

        self.model = model

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, prepared_data, epochs=100, batch_size=32):
        """
        Train the model

        Args:
            prepared_data (dict): Prepared data splits
            epochs (int): Number of epochs
            batch_size (int): Batch size
        """
        train_data = prepared_data['train']
        val_data = prepared_data.get('val', None)

        # Build model
        input_shape = (train_data['X'].shape[1], train_data['X'].shape[2])
        self.build_model(input_shape)

        print("Model Architecture:")
        self.model.summary()

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy' if val_data else 'accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_data else 'loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            # Update the ModelCheckpoint to save in Google Drive
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(OUTPUT_DIR, 'best_cough_classifier.h5'),  # Save to Drive
                monitor='val_accuracy' if val_data else 'accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train
        validation_data = (val_data['X'], val_data['y']) if val_data else None

        self.history = self.model.fit(
            train_data['X'], train_data['y'],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def save_training_plots(self):
        """Save training plots to Google Drive"""
        if self.history is None:
            return

        # Create plots just save
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plots to Google Drive
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        # Save history as numpy file
        np.save(os.path.join(OUTPUT_DIR, 'training_history.npy'), self.history.history)
        print(f"Training plots saved to {OUTPUT_DIR}")

    def evaluate_model(self, prepared_data):
        """Evaluate model on test data"""
        if 'test' not in prepared_data:
            print("No test data available")
            return

        test_data = prepared_data['test']

        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(
            test_data['X'], test_data['y'], verbose=0
        )

        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Predictions for detailed metrics
        y_pred = self.model.predict(test_data['X'])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(test_data['y'], axis=1)

        # Classification report
        report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=self.label_encoder.classes_
        )
        print("\nClassification Report:")
        print(report)

        # Save classification report to file
        with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        # Confusion Matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save confusion matrix to Google Drive
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print(f"Confusion matrix saved to {OUTPUT_DIR}")

        return test_accuracy

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def predict_single(self, audio_data):
        """Make prediction on single audio sample"""
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1, 1)
        elif len(audio_data.shape) == 2:
            audio_data = audio_data.reshape(1, audio_data.shape[0], audio_data.shape[1])

        prediction = self.model.predict(audio_data)
        predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)

        return predicted_class, confidence

    # Also save training history and plots
    def save_training_plots(self):
        """Save training plots to Google Drive"""
        if self.history is None:
            return

        # Create plots just save
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plots to Google Drive
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        # Save history as numpy file
        np.save(os.path.join(OUTPUT_DIR, 'training_history.npy'), self.history.history)
        print(f"Training plots saved to {OUTPUT_DIR}")

# Main execution
if __name__ == "__main__":
    # Initialize classifier
    classifier = CoughCNN()

    # Load preprocessed data
    data_path = "/content/extracted_data/processed_output"
    print("Loading preprocessed data...")
    data_splits = classifier.load_preprocessed_data(data_path)

    # Prepare data for training
    print("\nPreparing data...")
    prepared_data = classifier.prepare_data(data_splits)

    # Train model
    print("\nTraining model...")
    history = classifier.train_model(prepared_data, epochs=100, batch_size=32)

    # Save training plots
    classifier.save_training_plots()

    # Evaluate on test set
    test_accuracy = classifier.evaluate_model(prepared_data)

    # Save model summary to file
    with open(os.path.join(OUTPUT_DIR, 'model_summary.txt'), 'w') as f:
        classifier.model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Save final results summary
    with open(os.path.join(OUTPUT_DIR, 'final_results.txt'), 'w') as f:
        f.write("Cough Classification Model Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Final Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Number of epochs trained: {len(history.history['accuracy'])}\n")
        f.write(f"Classes: {', '.join(classifier.label_encoder.classes_)}\n")
        f.write(f"Training completed successfully!\n")

    print(f"\n🎉 Training completed! All outputs saved to: {OUTPUT_DIR}")
    print("📁 Files saved:")
    print("   - best_cough_classifier.h5 (trained model)")
    print("   - training_history.png (training plots)")
    print("   - training_history.npy (training history data)")
    print("   - confusion_matrix.png (confusion matrix)")
    print("   - classification_report.txt (detailed metrics)")
    print("   - model_summary.txt (model architecture)")
    print("   - final_results.txt (summary)")