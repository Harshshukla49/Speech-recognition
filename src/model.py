"""
Deep Learning Model Architectures for Speech Emotion Recognition
"""
import tensorflow as tf
from keras import layers, models, optimizers
from keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    LSTM, BatchNormalization, Activation,
    TimeDistributed, Reshape, Input, RepeatVector, Permute, Multiply, Lambda
)
import config


def create_cnn_model(input_shape=None, num_classes=None):
    """
    Create CNN model for emotion recognition
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of emotion classes
    
    Returns:
        Compiled CNN model
    """
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    model = models.Sequential([
        # First Convolutional Block
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(input_shape=None, num_classes=None):
    """
    Create LSTM model for emotion recognition
    
    Args:
        input_shape: Input shape (timesteps, features)
        num_classes: Number of emotion classes
    
    Returns:
        Compiled LSTM model
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    # Reshape input for LSTM
    if input_shape is None:
        input_shape = (config.INPUT_SHAPE[0], config.INPUT_SHAPE[1])
    
    model = models.Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_cnn_lstm_model(input_shape=None, num_classes=None):
    """
    Create hybrid CNN-LSTM model for emotion recognition
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of emotion classes
    
    Returns:
        Compiled CNN-LSTM model
    """
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    model = models.Sequential([
        # CNN Layers for feature extraction
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Reshape for LSTM
        Reshape((16, -1)),  # Reshape to (timesteps, features)
        
        # LSTM Layers for temporal modeling
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Dense Layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_attention_cnn_lstm_model(input_shape=None, num_classes=None):
    """
    Create CNN-LSTM model with attention mechanism
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of emotion classes
    
    Returns:
        Compiled CNN-LSTM model with attention
    """
    if input_shape is None:
        input_shape = config.INPUT_SHAPE
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN Layers
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Reshape for LSTM
    x = Reshape((16, -1))(x)
    
    # LSTM Layers
    lstm_out = LSTM(128, return_sequences=True)(x)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)

    # Apply attention
    attended = Multiply()([lstm_out, attention])
    attended = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)
    
    # Dense Layers
    x = Dense(256, activation='relu')(attended)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model(model_type='cnn_lstm', input_shape=None, num_classes=None):
    """
    Get model by type
    
    Args:
        model_type: Type of model ('cnn', 'lstm', 'cnn_lstm', 'attention_cnn_lstm')
        input_shape: Input shape
        num_classes: Number of classes
    
    Returns:
        Compiled model
    """
    model_dict = {
        'cnn': create_cnn_model,
        'lstm': create_lstm_model,
        'cnn_lstm': create_cnn_lstm_model,
        'attention_cnn_lstm': create_attention_cnn_lstm_model
    }
    
    if model_type not in model_dict:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model_dict[model_type](input_shape, num_classes)
    
    print(f"✅ Created {model_type.upper()} model")
    print(f"Total parameters: {model.count_params():,}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Model Architecture Module")
    print("=" * 50)
    
    # Create and display model summary
    model = get_model('cnn_lstm')
    print("\nModel Summary:")
    model.summary()
