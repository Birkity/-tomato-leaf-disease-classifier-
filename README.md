
# Tomato Leaf Disease Classifier

A deep learning-based system that uses Convolutional Neural Networks (CNN) to detect and classify diseases in tomato plant leaves. The system employs a two-stage classification approach for accurate disease identification.

## Features

- Two-stage classification process:
  1. Binary classification to identify tomato leaves
  2. Disease-specific classification for identified tomato leaves
- Support for 10 different tomato leaf conditions
- Real-time prediction capabilities
- RESTful API interface using FastAPI
- High accuracy disease classification
- Pre-trained models for immediate use

## Supported Classifications

### Tomato Diseases
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Healthy Leaves

## Technology Stack

- TensorFlow 2.x
- Python 3.10
- FastAPI
- NumPy
- Matplotlib
- PlantVillage Dataset

## Project Structure

```
tomato-leaf-disease-classifier/
├── api/
│   └── main.py           # FastAPI application
├── training/
│   ├── CNN_tomato.ipynb  # Model training notebook
│   └── CNN.ipynb         # Additional training utilities
├── saved-models/
│   ├── binary_model.keras    # Binary classification model
│   └── disease_model.keras   # Disease classification model
└── PlantVillage/         # Dataset directory
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tomato-leaf-disease-classifier.git
cd tomato-leaf-disease-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models (optional):
```bash
# Place downloaded models in saved-models/ directory
```

## Usage

### API Service

1. Start the FastAPI server:
```bash
uvicorn api.main:app --reload
```

2. Access the API at `http://localhost:8000`

3. Use the `/predict` endpoint to classify images:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

### Training New Models

1. Open `training/CNN_tomato.ipynb` in Jupyter Notebook
2. Follow the notebook cells to:
   - Prepare the dataset
   - Train the binary classifier
   - Train the disease classifier
   - Save the models

## Model Architecture

- CNN architecture with multiple convolutional and pooling layers
- Binary classification model for tomato leaf detection
- Multi-class classification model for disease identification
- Trained on the PlantVillage dataset

## Performance

- Binary Classification Accuracy: ~99%
- Disease Classification Accuracy: ~95%
- Real-time prediction capability: <1 second per image

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PlantVillage Dataset for providing the training data
- TensorFlow team for the deep learning framework
- FastAPI for the web framework
```
