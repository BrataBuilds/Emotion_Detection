import modal

app = modal.App("Emotion-Recognition")

# packages = [
#     "tensorflow",
#     "opencv-python-headless",
#     "pillow",
#     "numpy"
# ]
img = (modal.Image.debian_slim().pip_install("tensorflow",
    "pillow",
    "numpy")
    .add_local_dir("data/test",remote_path="/root/test")
    .add_local_dir("data/train",remote_path="/root/train")
    .run_commands(
             # Attempted to resolve an issue with installing CUDA dependencies
             "apt-get update",
             "apt-get install -y nvidia-cuda-toolkit"))

@app.function(image=img, gpu="T4",timeout=3600) # Earlier run got interrupted by timeout.
def train_emotion_model():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_data_gen = ImageDataGenerator(rescale=1./255)
    validation_data_gen = ImageDataGenerator(rescale=1./255)

    train_dir = "/root/train"  
    test_dir = "/root/test"

    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
    )

    validation_generator = validation_data_gen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
    )
# CNN Implementation
    emotion_model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    emotion_model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64
    )

# TODO: Implement the model architecture and weights to be saved locally
    # Save on Modal first
    emotion_model.save_weights('/root/emotion_model.h5')
    
    return 0

@app.local_entrypoint()
def main():

if __name__ == "__main__":
    modal.runner.main(app)