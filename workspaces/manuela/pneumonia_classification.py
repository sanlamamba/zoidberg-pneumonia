import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

train_dir = '../../data/train'
val_dir = '../../data/val'  
test_dir = '../../data/test'  

def display_sample_images(folder, num_images=5):
    categories = ['NORMAL', 'PNEUMONIA']
    for category in categories:
        path = os.path.join(folder, category)
        images = os.listdir(path)
        for img_name in images[:num_images]:
            img_path = os.path.join(path, img_name)
            img = plt.imread(img_path)
            plt.imshow(img, cmap='gray')
            plt.title(category)
            plt.show()

display_sample_images(train_dir)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

def load_data(folder):
    data = []
    labels = []
    categories = ['NORMAL', 'PNEUMONIA']
    for category in categories:
        path = os.path.join(folder, category)
        class_num = categories.index(category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (150, 150))
                data.append(img)
                labels.append(class_num)
            except Exception as e:
                print(e)
    data = np.array(data).reshape(-1, 150 * 150)
    labels = np.array(labels)
    return data, labels

train_data, train_labels = load_data(train_dir)
val_data, val_labels = load_data(val_dir)

svm = SVC(kernel='linear')
svm.fit(train_data, train_labels)
svm_preds = svm.predict(val_data)

print(f'Accuracy du SVM: {accuracy_score(val_labels, svm_preds):.2f}')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10
)

cross_val_scores = cross_val_score(SVC(kernel='linear'), train_data, train_labels, cv=5)
print(f'Scores de validation croisée: {cross_val_scores}')
print(f'Moyenne des scores de validation croisée: {np.mean(cross_val_scores):.2f}')

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(train_data, train_labels)

print(f'Best Parameters: {grid.best_params_}')
print(f'Best Score: {grid.best_score_:.2f}')

pca = PCA(n_components=50)
train_data_pca = pca.fit_transform(train_data)
val_data_pca = pca.transform(val_data)

svm_pca = SVC(kernel='linear')
svm_pca.fit(train_data_pca, train_labels)
svm_pca_preds = svm_pca.predict(val_data_pca)

print(f'Accuracy du SVM avec PCA: {accuracy_score(val_labels, svm_pca_preds):.2f}')

val_preds = model.predict(val_generator)
roc_auc = roc_auc_score(val_generator.classes, val_preds)
print(f'ROC-AUC Score pour le CNN: {roc_auc:.2f}')

fpr, tpr, _ = roc_curve(val_generator.classes, val_preds)
plt.plot(fpr, tpr, marker='.', label='CNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

model.save('cnn_pneumonia_model.h5')

from tensorflow.keras.models import load_model
model = load_model('cnn_pneumonia_model.h5')
