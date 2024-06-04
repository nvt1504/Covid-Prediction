import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

# Đường dẫn tới thư mục chứa dữ liệu huấn luyện và kiểm tra
train_dir = 'Data/Train'
validation_dir = 'Data/validation'

# Chuẩn bị dữ liệu và chuẩn hóa
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Chuẩn hóa dữ liệu
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)  # Chuẩn hóa dữ liệu

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary'
)

# Tải mô hình VGG19 đã được huấn luyện trước
base_model = VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))

# Đóng băng các lớp của mô hình VGG19
for layer in base_model.layers:
    layer.trainable = False

# Thêm lớp chuyển đổi ảnh từ 1 kênh thành 3 kênh
input = Input(shape=(256, 256, 1))
x = Conv2D(3, (3, 3), padding='same')(input)  # Chuyển đổi từ 1 kênh sang 3 kênh

# Kết hợp với mô hình VGG19
x = base_model(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Tạo mô hình hoàn chỉnh
model = Model(inputs=input, outputs=predictions)

# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Lưu mô hình
model.save('covid19_vgg19_model.h5')

# Đánh giá mô hình
loss, accuracy = model.evaluate(validation_generator)
print(f'Model accuracy on validation set: {accuracy * 100:.2f}%')