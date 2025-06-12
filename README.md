# TomatoQAS (Tomato Quality Assessment System)

A deep learning-based system for assessing tomato quality using computer vision. This project was developed as part of a university Project Based Learning course.

## Overview

TomatoQAS is a machine learning system that uses the VGG16 architecture to classify and assess the quality of tomatoes based on visual characteristics. The system can process both individual images and batches of images for quality assessment.

## Dataset

The project uses a custom dataset available on Kaggle:
- [TomatoQAS Dataset](https://www.kaggle.com/datasets/cronenberg64/tomatoqas-dataset)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/TomatoQAS.git
cd TomatoQAS
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `VGG16.py`: Main training script for the model
- `test_dir.py`: Script for batch testing on directories
- `test_img.py`: Script for testing individual images
- `requirements.txt`: Project dependencies
- `poster/`: Contains project presentation materials

## Usage

### Training the Model

1. Download the dataset from the provided Kaggle link
2. Adjust the data paths in `VGG16.py` to match your local directory structure
3. Run the training script:
```bash
python VGG16.py
```

### Testing

#### Batch Testing
To test the model on a directory of images:
1. Modify the directory path in `test_dir.py`
2. Run:
```bash
python test_dir.py
```

#### Single Image Testing
To test the model on a single image:
1. Update the image path in `test_img.py`
2. Run:
```bash
python test_img.py
```

## Documentation

Additional project documentation and presentation materials are available in the `poster/` directory.

## Contact

For any inquiries or questions about the project, please feel free to reach out to the author.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*This project was developed as part of a university Project Based Learning course.*



