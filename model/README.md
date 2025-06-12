# Model Setup Instructions

1. Place your trained model file (`tomato_model.h5`) in this directory
2. The model should be a TensorFlow/Keras model that:
   - Takes input images of size 224x224 pixels
   - Outputs predictions for tomato quality (Good/Bad)
   - Uses the same preprocessing steps as defined in `app.py`

## Model Requirements

- Input shape: (224, 224, 3)
- Output: Binary classification (Good/Bad)
- Preprocessing: Images should be normalized to [0,1] range 