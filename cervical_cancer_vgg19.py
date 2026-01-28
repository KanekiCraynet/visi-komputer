# ============================================================================
# DETEKSI DINI KANKER SERVIKS DENGAN CNN BERBASIS VGG19 (ADVANCED VERSION)
# Studi Kasus UAS Visi Komputer
# Implementasi di Google Colab dengan Transfer Learning
# ============================================================================
# FITUR LANJUTAN:
# - Class Weighting untuk imbalanced data
# - Learning Rate Scheduling (Cosine Annealing)
# - Label Smoothing
# - Mixed Precision Training
# - GradCAM Visualization
# - ROC Curves dan AUC per class
# - K-Fold Cross Validation (optional)
# - TensorBoard Logging
# - Early Stopping dengan Patience Adaptive
# ============================================================================

# ============================================================================
# BAGIAN 1: IMPORT LIBRARY
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, LearningRateScheduler, CSVLogger
)
from tensorflow.keras.regularizers import l2
import os
import time
from datetime import datetime
import warnings
import cv2
from collections import Counter
warnings.filterwarnings('ignore')

# Enable Mixed Precision for faster training (if GPU supports it)
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed Precision: Enabled (float16)")
except:
    print("Mixed Precision: Not available, using float32")

print("="*60)
print("TensorFlow Version:", tf.__version__)
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU Name:", tf.config.list_physical_devices('GPU')[0].name)
print("="*60)

# ============================================================================
# BAGIAN 2: KONFIGURASI DAN HYPERPARAMETERS
# ============================================================================

class Config:
    """Konfigurasi terpusat untuk eksperimen"""
    # Data
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 5
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Training Phase 1 (Feature Extraction)
    EPOCHS_PHASE1 = 30
    LEARNING_RATE_PHASE1 = 0.001
    
    # Training Phase 2 (Fine-tuning)
    EPOCHS_PHASE2 = 30
    LEARNING_RATE_PHASE2 = 0.0001
    
    # Training Phase 3 (Deep Fine-tuning) - NEW!
    EPOCHS_PHASE3 = 20
    LEARNING_RATE_PHASE3 = 0.00001
    
    # Regularization
    DROPOUT_RATE = 0.5
    L2_REGULARIZATION = 0.001
    LABEL_SMOOTHING = 0.1
    
    # Callbacks
    EARLY_STOP_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    
    # Augmentation
    ROTATION_RANGE = 40
    WIDTH_SHIFT = 0.2
    HEIGHT_SHIFT = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.3
    BRIGHTNESS_RANGE = [0.7, 1.3]
    
    # Paths
    DATASET_PATH = './dataset/'
    OUTPUT_DIR = './outputs/'
    
    # Class Names
    CLASS_NAMES = [
        'Superficial-Intermediate',  # Normal
        'Parabasal',                  # Normal/Atrophy
        'Koilocytotic',              # Abnormal (HPV)
        'Dyskeratotic',              # Abnormal
        'Metaplastic'                # Benign
    ]

config = Config()

# Create output directory
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("\nEXPERIMENT CONFIGURATION")
print("="*60)
for attr in dir(config):
    if not attr.startswith('_'):
        print(f"  {attr}: {getattr(config, attr)}")

# ============================================================================
# BAGIAN 3: DOWNLOAD DATASET (KAGGLEHUB)
# ============================================================================

print("\n[STEP 0] Downloading Dataset...")

try:
    import kagglehub
    DATASET_PATH = kagglehub.dataset_download("prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed")
    print(f"Dataset downloaded to: {DATASET_PATH}")
except Exception as e:
    print(f"Kaggle download failed: {e}")
    print("Using manual path...")
    DATASET_PATH = config.DATASET_PATH

# ============================================================================
# BAGIAN 4: CUSTOM LEARNING RATE SCHEDULERS
# ============================================================================

def cosine_annealing_schedule(epoch, lr, total_epochs, min_lr=1e-7):
    """Cosine Annealing Learning Rate Schedule"""
    return min_lr + (lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs)) / 2

def warmup_cosine_schedule(epoch, initial_lr, warmup_epochs=5, total_epochs=50):
    """Warmup + Cosine Annealing"""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return initial_lr * (1 + np.cos(np.pi * progress)) / 2

class WarmUpCosineDecay(keras.callbacks.Callback):
    """Custom callback untuk Warmup + Cosine Decay"""
    def __init__(self, initial_lr, warmup_epochs, total_epochs, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.lr_history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress)) / 2
        
        K.set_value(self.model.optimizer.learning_rate, lr)
        self.lr_history.append(lr)
        print(f"\nEpoch {epoch+1}: Learning Rate = {lr:.2e}")

# ============================================================================
# BAGIAN 5: ADVANCED DATA AUGMENTATION
# ============================================================================

print("\n[STEP 1] Initializing Advanced Data Augmentation...")

def get_advanced_augmentation():
    """Advanced augmentation dengan lebih banyak variasi"""
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT,
        height_shift_range=config.HEIGHT_SHIFT,
        shear_range=config.SHEAR_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=config.BRIGHTNESS_RANGE,
        channel_shift_range=30,
        fill_mode='reflect',  # Better than 'nearest' for medical images
        preprocessing_function=lambda x: add_noise(x, noise_factor=0.05)
    )

def add_noise(image, noise_factor=0.05):
    """Add random Gaussian noise"""
    if np.random.random() > 0.5:
        noise = np.random.normal(0, noise_factor, image.shape)
        return np.clip(image + noise * 255, 0, 255)
    return image

# Data Generators
train_datagen = get_advanced_augmentation()
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train') if os.path.exists(os.path.join(DATASET_PATH, 'train')) else DATASET_PATH,
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'validation') if os.path.exists(os.path.join(DATASET_PATH, 'validation')) else DATASET_PATH,
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test') if os.path.exists(os.path.join(DATASET_PATH, 'test')) else DATASET_PATH,
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nDataset Statistics:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {validation_generator.samples}")
print(f"  Test samples: {test_generator.samples}")

# ============================================================================
# BAGIAN 6: COMPUTE CLASS WEIGHTS (UNTUK IMBALANCED DATA)
# ============================================================================

print("\n[STEP 2] Computing Class Weights for Imbalanced Data...")

# Get class distribution
class_counts = Counter(train_generator.classes)
total_samples = sum(class_counts.values())

print("\nClass Distribution:")
for cls_idx, count in sorted(class_counts.items()):
    cls_name = list(train_generator.class_indices.keys())[cls_idx]
    percentage = count / total_samples * 100
    print(f"  {cls_name}: {count} samples ({percentage:.1f}%)")

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))

print("\n⚖️ Class Weights:")
for cls_idx, weight in class_weight_dict.items():
    cls_name = list(train_generator.class_indices.keys())[cls_idx]
    print(f"  {cls_name}: {weight:.4f}")

# ============================================================================
# BAGIAN 7: BUILD ADVANCED MODEL ARCHITECTURE
# ============================================================================

print("\n[STEP 3] Building Advanced Model Architecture...")

def build_advanced_model(num_classes, l2_reg=0.001, dropout_rate=0.5):
    """Build VGG19 model dengan regularisasi dan arsitektur yang lebih baik"""
    
    # Load VGG19 base
    base_model = VGG19(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Input layer
    inputs = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    
    # Data augmentation layers (applied during training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # VGG19 base
    x = base_model(x, training=False)
    
    # Global Average Pooling (better than Flatten for generalization)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense layers with regularization
    x = layers.Dense(512, kernel_regularizer=l2(l2_reg), name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, kernel_regularizer=l2(l2_reg), name='fc2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.6)(x)
    
    x = layers.Dense(128, kernel_regularizer=l2(l2_reg), name='fc3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.4)(x)
    
    # Output layer dengan float32 untuk numerical stability
    x = layers.Dense(num_classes, kernel_regularizer=l2(l2_reg), name='pre_output')(x)
    outputs = layers.Activation('softmax', dtype='float32', name='output')(x)
    
    model = keras.Model(inputs, outputs, name='VGG19_CervicalCancer_Advanced')
    
    return model, base_model

model, base_model = build_advanced_model(
    config.NUM_CLASSES, 
    l2_reg=config.L2_REGULARIZATION,
    dropout_rate=config.DROPOUT_RATE
)

print("\n📋 Model Architecture Summary:")
model.summary()

# ============================================================================
# BAGIAN 8: CUSTOM LOSS DENGAN LABEL SMOOTHING
# ============================================================================

print("\n[STEP 4] Setting up Loss Function with Label Smoothing...")

def get_loss_function(label_smoothing=0.1):
    """Categorical Crossentropy dengan Label Smoothing"""
    return keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing,
        from_logits=False
    )

loss_fn = get_loss_function(config.LABEL_SMOOTHING)
print(f"  Label Smoothing: {config.LABEL_SMOOTHING}")

# ============================================================================
# BAGIAN 9: COMPREHENSIVE METRICS
# ============================================================================

def get_metrics():
    """Get comprehensive metrics for multi-class classification"""
    return [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc', multi_label=True),
        keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')
    ]

# ============================================================================
# BAGIAN 10: CALLBACKS SETUP
# ============================================================================

print("\n[STEP 5] Setting up Callbacks...")

def get_callbacks(phase, learning_rate, epochs, output_dir):
    """Get callbacks untuk setiap phase"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Early Stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        
        # Model Checkpoint
        ModelCheckpoint(
            filepath=os.path.join(output_dir, f'best_model_phase{phase}.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce LR on Plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-8,
            verbose=1
        ),
        
        # CSV Logger
        CSVLogger(
            os.path.join(output_dir, f'training_log_phase{phase}.csv'),
            append=False
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs', f'phase{phase}_{timestamp}'),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ),
        
        # Custom LR Schedule
        WarmUpCosineDecay(
            initial_lr=learning_rate,
            warmup_epochs=3,
            total_epochs=epochs
        )
    ]
    
    return callbacks

# ============================================================================
# BAGIAN 11: TRAINING PHASE 1 - FEATURE EXTRACTION
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: FEATURE EXTRACTION (Frozen VGG19 Base)")
print("="*70)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE_PHASE1),
    loss=loss_fn,
    metrics=get_metrics()
)

# Training Phase 1
callbacks_p1 = get_callbacks(1, config.LEARNING_RATE_PHASE1, config.EPOCHS_PHASE1, config.OUTPUT_DIR)

start_time = time.time()
history_phase1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=config.EPOCHS_PHASE1,
    callbacks=callbacks_p1,
    class_weight=class_weight_dict,
    verbose=1
)
training_time_phase1 = time.time() - start_time

print(f"\nPhase 1 completed in {training_time_phase1:.1f} seconds")
print(f"   Best Validation Accuracy: {max(history_phase1.history['val_accuracy']):.4f}")

# ============================================================================
# BAGIAN 12: TRAINING PHASE 2 - FINE-TUNING (LAST 2 BLOCKS)
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: FINE-TUNING (Last 2 Convolutional Blocks)")
print("="*70)

# Unfreeze last 2 blocks
base_model.trainable = True
for layer in base_model.layers[:-8]:  # Freeze all except last 8 layers
    layer.trainable = False

print(f"\nTrainable layers: {sum([1 for l in base_model.layers if l.trainable])}")

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE_PHASE2),
    loss=loss_fn,
    metrics=get_metrics()
)

# Training Phase 2
callbacks_p2 = get_callbacks(2, config.LEARNING_RATE_PHASE2, config.EPOCHS_PHASE2, config.OUTPUT_DIR)

start_time = time.time()
history_phase2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=config.EPOCHS_PHASE2,
    callbacks=callbacks_p2,
    class_weight=class_weight_dict,
    verbose=1
)
training_time_phase2 = time.time() - start_time

print(f"\nPhase 2 completed in {training_time_phase2:.1f} seconds")
print(f"   Best Validation Accuracy: {max(history_phase2.history['val_accuracy']):.4f}")

# ============================================================================
# BAGIAN 13: TRAINING PHASE 3 - DEEP FINE-TUNING (ALL LAYERS)
# ============================================================================

print("\n" + "="*70)
print("🔬 PHASE 3: DEEP FINE-TUNING (All Layers Trainable)")
print("="*70)

# Unfreeze ALL layers
base_model.trainable = True

print(f"\nAll {len(base_model.layers)} layers are now trainable")

# Recompile with very low learning rate
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE_PHASE3),
    loss=loss_fn,
    metrics=get_metrics()
)

# Training Phase 3
callbacks_p3 = get_callbacks(3, config.LEARNING_RATE_PHASE3, config.EPOCHS_PHASE3, config.OUTPUT_DIR)

start_time = time.time()
history_phase3 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=config.EPOCHS_PHASE3,
    callbacks=callbacks_p3,
    class_weight=class_weight_dict,
    verbose=1
)
training_time_phase3 = time.time() - start_time

print(f"\nPhase 3 completed in {training_time_phase3:.1f} seconds")
print(f"   Best Validation Accuracy: {max(history_phase3.history['val_accuracy']):.4f}")

# ============================================================================
# BAGIAN 14: PLOT COMPREHENSIVE TRAINING HISTORY
# ============================================================================

print("\n[STEP 6] Plotting Comprehensive Training History...")

def plot_training_history(histories, phases, output_dir):
    """Plot training history dari semua phase"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Combine histories
    metrics = ['accuracy', 'loss', 'precision', 'recall', 'auc']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    all_train_acc = []
    all_val_acc = []
    all_train_loss = []
    all_val_loss = []
    phase_boundaries = [0]
    
    for h in histories:
        all_train_acc.extend(h.history['accuracy'])
        all_val_acc.extend(h.history['val_accuracy'])
        all_train_loss.extend(h.history['loss'])
        all_val_loss.extend(h.history['val_loss'])
        phase_boundaries.append(len(all_train_acc))
    
    epochs = range(1, len(all_train_acc) + 1)
    
    # Plot 1: Accuracy
    axes[0, 0].plot(epochs, all_train_acc, 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs, all_val_acc, 'r-', label='Validation', linewidth=2)
    for i, boundary in enumerate(phase_boundaries[1:-1]):
        axes[0, 0].axvline(boundary, color='gray', linestyle='--', alpha=0.7)
        axes[0, 0].text(boundary, max(all_val_acc)*0.95, f'P{i+2}', fontsize=10)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss
    axes[0, 1].plot(epochs, all_train_loss, 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs, all_val_loss, 'r-', label='Validation', linewidth=2)
    for i, boundary in enumerate(phase_boundaries[1:-1]):
        axes[0, 1].axvline(boundary, color='gray', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate (if available)
    lr_history = []
    for cb in callbacks_p1 + callbacks_p2 + callbacks_p3:
        if isinstance(cb, WarmUpCosineDecay):
            lr_history.extend(cb.lr_history)
    
    if lr_history:
        axes[0, 2].plot(range(1, len(lr_history)+1), lr_history, 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Precision
    all_precision = []
    all_val_precision = []
    for h in histories:
        all_precision.extend(h.history.get('precision', []))
        all_val_precision.extend(h.history.get('val_precision', []))
    
    if all_precision:
        axes[1, 0].plot(epochs, all_precision, 'b-', label='Training', linewidth=2)
        axes[1, 0].plot(epochs, all_val_precision, 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Recall
    all_recall = []
    all_val_recall = []
    for h in histories:
        all_recall.extend(h.history.get('recall', []))
        all_val_recall.extend(h.history.get('val_recall', []))
    
    if all_recall:
        axes[1, 1].plot(epochs, all_recall, 'b-', label='Training', linewidth=2)
        axes[1, 1].plot(epochs, all_val_recall, 'r-', label='Validation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Phase Comparison
    phase_results = []
    for i, h in enumerate(histories):
        phase_results.append({
            'Phase': f'Phase {i+1}',
            'Best Val Acc': max(h.history['val_accuracy']),
            'Final Val Acc': h.history['val_accuracy'][-1],
            'Epochs': len(h.history['accuracy'])
        })
    
    phases_df = pd.DataFrame(phase_results)
    x_pos = range(len(phases_df))
    axes[1, 2].bar(x_pos, phases_df['Best Val Acc'], color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[1, 2].set_xlabel('Training Phase')
    axes[1, 2].set_ylabel('Best Validation Accuracy')
    axes[1, 2].set_title('Phase Comparison', fontsize=12, fontweight='bold')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(phases_df['Phase'])
    for i, v in enumerate(phases_df['Best Val Acc']):
        axes[1, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_comprehensive.png'), dpi=150, bbox_inches='tight')
    plt.show()

plot_training_history(
    [history_phase1, history_phase2, history_phase3],
    ['Phase 1', 'Phase 2', 'Phase 3'],
    config.OUTPUT_DIR
)

# ============================================================================
# BAGIAN 15: EVALUASI MODEL
# ============================================================================

print("\n[STEP 7] Evaluating Model on Test Set...")

# Load best model
best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model_phase3.keras')
if os.path.exists(best_model_path):
    model = keras.models.load_model(best_model_path)
    print(f"Loaded best model from Phase 3")

# Evaluate
results = model.evaluate(test_generator, verbose=1)
metrics_names = model.metrics_names

print(f"\n{'='*60}")
print(f"📊 TEST SET EVALUATION RESULTS")
print(f"{'='*60}")
for name, value in zip(metrics_names, results):
    print(f"  {name.upper()}: {value:.4f}")
print(f"{'='*60}")

# ============================================================================
# BAGIAN 16: PREDICTIONS DAN ADVANCED ANALYSIS
# ============================================================================

print("\n[STEP 8] Generating Predictions...")

# Get predictions
test_generator.reset()
y_pred_proba = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = test_generator.classes

# Class names dari generator
CLASS_NAMES = list(test_generator.class_indices.keys())

# ============================================================================
# BAGIAN 17: ROC CURVES PER CLASS
# ============================================================================

print("\n[STEP 9] Generating ROC Curves...")

def plot_roc_curves(y_true, y_pred_proba, class_names, output_dir):
    """Plot ROC curve untuk setiap kelas"""
    
    n_classes = len(class_names)
    
    # Binarize labels
    y_true_bin = keras.utils.to_categorical(y_true, num_classes=n_classes)
    
    # Compute ROC curve dan AUC untuk setiap kelas
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-class ROC
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        axes[0].plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves per Class', fontsize=12, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # AUC Bar Chart
    auc_values = [roc_auc[i] for i in range(n_classes)]
    bars = axes[1].bar(class_names, auc_values, color=colors, edgecolor='black')
    axes[1].axhline(y=np.mean(auc_values), color='red', linestyle='--', 
                    label=f'Mean AUC: {np.mean(auc_values):.3f}')
    axes[1].set_ylabel('AUC Score')
    axes[1].set_title('AUC Scores per Class', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar, auc_val in zip(bars, auc_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{auc_val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return roc_auc

roc_auc_scores = plot_roc_curves(y_true, y_pred_proba, CLASS_NAMES, config.OUTPUT_DIR)

# ============================================================================
# BAGIAN 18: CONFUSION MATRIX (ENHANCED)
# ============================================================================

print("\n[STEP 10] Generating Enhanced Confusion Matrix...")

def plot_enhanced_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Enhanced confusion matrix dengan normalisasi"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Normalized (percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage'})
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_enhanced.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return cm

cm = plot_enhanced_confusion_matrix(y_true, y_pred, CLASS_NAMES, config.OUTPUT_DIR)

# ============================================================================
# BAGIAN 19: GRADCAM VISUALIZATION
# ============================================================================

print("\n[STEP 11] Generating GradCAM Visualizations...")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate GradCAM heatmap"""
    
    # Create model that maps input to conv layer output and predictions
    grad_model = keras.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Gradient of the predicted class with respect to conv output
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Mean intensity of gradient over specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv outputs by gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def visualize_gradcam(model, test_generator, class_names, output_dir, num_samples=5):
    """Visualize GradCAM untuk beberapa sample"""
    
    # Get last conv layer name
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer.name
            break
    
    # Get samples
    test_generator.reset()
    images, labels = next(test_generator)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        true_label = np.argmax(labels[i])
        
        # Prediction
        pred = model.predict(np.expand_dims(img, 0), verbose=0)
        pred_label = np.argmax(pred[0])
        confidence = pred[0][pred_label]
        
        # Generate heatmap
        try:
            heatmap = make_gradcam_heatmap(np.expand_dims(img, 0), model, last_conv_layer)
            
            # Resize heatmap
            heatmap = cv2.resize(heatmap, (config.IMG_SIZE, config.IMG_SIZE))
            heatmap = np.uint8(255 * heatmap)
            
            # Apply colormap
            jet = plt.cm.jet
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]
            
            # Superimpose
            superimposed = jet_heatmap * 0.4 + img
            superimposed = superimposed / superimposed.max()
        except:
            superimposed = img
            heatmap = np.zeros((config.IMG_SIZE, config.IMG_SIZE))
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\nTrue: {class_names[true_label]}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title('GradCAM Heatmap')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(superimposed)
        axes[i, 2].set_title(f'Overlay\nPred: {class_names[pred_label]} ({confidence:.2%})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam_visualization.png'), dpi=150, bbox_inches='tight')
    plt.show()

try:
    visualize_gradcam(model, test_generator, CLASS_NAMES, config.OUTPUT_DIR)
except Exception as e:
    print(f"GradCAM visualization skipped: {e}")

# ============================================================================
# BAGIAN 20: CLASSIFICATION REPORT
# ============================================================================

print("\n[STEP 12] Generating Classification Report...")

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print("\n" + report)

# Save report
with open(os.path.join(config.OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("="*70 + "\n")
    f.write("CLASSIFICATION REPORT - VGG19 Cervical Cancer Detection\n")
    f.write("="*70 + "\n\n")
    f.write(report)

# ============================================================================
# BAGIAN 21: SAVE FINAL MODEL
# ============================================================================

print("\n[STEP 13] Saving Final Model...")

# Save in multiple formats
model.save(os.path.join(config.OUTPUT_DIR, 'vgg19_cervical_cancer_final.keras'))
model.save(os.path.join(config.OUTPUT_DIR, 'vgg19_cervical_cancer_final.h5'))
model.save_weights(os.path.join(config.OUTPUT_DIR, 'vgg19_weights.weights.h5'))

# Save as SavedModel format (for TensorFlow Serving)
model.export(os.path.join(config.OUTPUT_DIR, 'saved_model'))

print("Model saved in multiple formats:")
print(f"   - {config.OUTPUT_DIR}vgg19_cervical_cancer_final.keras")
print(f"   - {config.OUTPUT_DIR}vgg19_cervical_cancer_final.h5")
print(f"   - {config.OUTPUT_DIR}saved_model/")

# ============================================================================
# BAGIAN 22: EXPERIMENT SUMMARY
# ============================================================================

print("\n" + "="*70)
print("EXPERIMENT SUMMARY")
print("="*70)

total_training_time = training_time_phase1 + training_time_phase2 + training_time_phase3

# Get best metrics
test_accuracy = results[metrics_names.index('accuracy')]
test_precision = results[metrics_names.index('precision')]
test_recall = results[metrics_names.index('recall')]
test_auc = results[metrics_names.index('auc')]

summary = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    EXPERIMENT SUMMARY REPORT                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  MODEL ARCHITECTURE                                                   ║
║    Base Model: VGG19 (ImageNet Pretrained)                           ║
║    Total Parameters: {model.count_params():,}                           
║    FC Layers: 512 → 256 → 128 → 5                                    ║
║                                                                       ║
║  TRAINING CONFIGURATION                                               ║
║    Image Size: {config.IMG_SIZE}×{config.IMG_SIZE}                                            
║    Batch Size: {config.BATCH_SIZE}                                               
║    Label Smoothing: {config.LABEL_SMOOTHING}                                        
║    L2 Regularization: {config.L2_REGULARIZATION}                                     
║                                                                       ║
║  TRAINING PHASES                                                      ║
║    Phase 1 (Feature Extraction): {len(history_phase1.history['accuracy'])} epochs, {training_time_phase1:.1f}s          
║    Phase 2 (Fine-tuning): {len(history_phase2.history['accuracy'])} epochs, {training_time_phase2:.1f}s                 
║    Phase 3 (Deep Fine-tuning): {len(history_phase3.history['accuracy'])} epochs, {training_time_phase3:.1f}s            
║    Total Time: {total_training_time:.1f} seconds                                  
║                                                                       ║
║  TEST SET RESULTS                                                     ║
║    Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)                               
║    Precision: {test_precision:.4f}                                            
║    Recall:    {test_recall:.4f}                                            
║    AUC:       {test_auc:.4f}                                            
║                                                                       ║
║  OUTPUT FILES                                                         ║
║    - training_history_comprehensive.png                              ║
║    - confusion_matrix_enhanced.png                                   ║
║    - roc_curves.png                                                  ║
║    - gradcam_visualization.png                                       ║
║    - classification_report.txt                                       ║
║    - vgg19_cervical_cancer_final.keras                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

print(summary)

# Save summary
with open(os.path.join(config.OUTPUT_DIR, 'experiment_summary.txt'), 'w') as f:
    f.write(summary)

print("\nTraining Complete! All outputs saved to:", config.OUTPUT_DIR)
