# LaneVision — Low-Light Lane Enhancement & Detection System

A Flask-based web application for enhancing and detecting lanes in low-light/thermal images using dual baseline enhancement models (Enhanced Retinex vs Standard YOLOv8) with real-time Hough line detection.

## 🎯 Features

- **Dual Baseline Models**:
  - **Enhanced Retinex**: Maximum brightness with aggressive tone mapping (75-85% detection confidence)
  - **Standard YOLOv8**: Balanced brightening for practical real-time performance (60-75% detection confidence)

- **Advanced Lane Detection**: Hough line detection with adaptive Canny edge detection
- **Detection Confidence Score**: Real-time confidence percentage based on detected lane lines
- **Interactive Parameter Controls**:
  - Contrast Factor slider (1-100) for gamma correction
  - Noise Reduction slider (0-10) for denoise strength
  - Baseline Model selector (Enhanced vs Standard)

- **Quality Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - Shannon Entropy analysis
  - Processing time measurement
  - Detection confidence scoring

- **Real-time Processing**: ~1 second per image
- **Drag & Drop Upload**: Easy image loading interface
- **Image Comparison**: Side-by-side original vs enhanced visualization
- **Lane Visualization**: Cyan-colored detected lane lines overlay

## 🏗️ Project Structure

```
Final Project/
├── algorithm.py                    # Image enhancement & lane detection algorithms
├── app.py                          # Flask backend with dual models & APIs
├── templates/
│   ├── index.html                  # Landing page
│   └── control.html                # Control panel with interactive UI
├── static/
│   ├── css/
│   │   └── style.css               # Dark mode responsive stylesheet
│   └── js/
│   |   ├── main.js                 # Landing page interactions
│   |   └── control.js              # Control panel & upload logic
|   └── uploads                     # Temporary processed image storage
|── Dockerfile                      # Docker container configuration
|── docker-compose.yml              # Multi-container Docker configuration
|── .dockerignore                   # Ignore unnecessary Docker build files
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📋 Technical Details

### Enhancement Algorithms

**Enhanced Retinex Pipeline**:
- Very aggressive gamma correction (gamma = 1.0 / (1.0 + eta * 0.08))
- Double CLAHE pass for cumulative brightness boost
- Strong denoise with h = 6-15
- Guided filter radius: 25, eps: 0.025
- Final contrast boost (+15%)

**Standard YOLOv8 Pipeline**:
- Conservative gamma correction (gamma = 1.0 / (1.0 + eta * 0.02))
- Light CLAHE (clipLimit 1.5) for subtle enhancement
- Moderate denoise with h = 3-10
- Guided filter radius: 12, eps: 0.055
- Reduced eta * 0.85 for balanced results

### Lane Detection

- Bilateral filtering for edge-preserving smoothing
- Adaptive CLAHE histogram equalization
- Adaptive Canny edge detection with dynamic thresholds
- Hough line detection:
  - Threshold: 20 (very sensitive)
  - Min line length: 20 pixels
  - Max gap: 30 pixels

### Batch Processing

+- Supports processing multiple uploaded images sequentially.
+- Each image is processed independently using the selected enhancement model.
+- Useful for dataset evaluation and benchmarking experiments.

### Performance Caching

+- Frequently processed images can be temporarily cached in memory.
+- Reduces repeated processing time for identical inputs.
+- Improves responsiveness during repeated testing.
+
### Docker Containerization

+- The application can run inside Docker containers.
+- Ensures consistent environment setup across different systems.
+- Supports easy deployment using Docker Compose.
**Confidence Calculation**:
```
confidence = min(98%, 25% + line_count*3.5 + avg_length/3)

```

## ⚙️ Installation

### Requirements
- Python 3.8+
- Flask
- OpenCV (cv2)
- NumPy

+### Docker Installation
+
+Run the application using Docker:
+
+```bash
+docker-compose up --build
+```
+
+Then open:
+
+```bash
+http://localhost:5000
+```


### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/nop4al/LaneVision.git
cd LaneVision
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Flask application**:
```bash
python app.py
```

4. **Open in browser**:
```
http://localhost:5000
```

## 🎮 Usage

### Landing Page (`/`)
- Introduction to LaneVision
- Feature overview
- Navigation to control panel

### Control Panel (`/control`)

1. **Select Baseline Model**:
   - "Enhanced Retinex" - For maximum brightness & contrast
   - "Standard YOLOv8" - For balanced, natural-looking results

2. **Adjust Parameters**:
   - **Contrast Factor**: 1-100 (higher = brighter)
   - **Noise Reduction**: 0-10 (higher = more smoothing)

3. **Upload Image**:
   - Drag & drop or click to select
   - Supported: PNG, JPG, JPEG, TIFF
   - Max size: 10 MB
   + Max Size: 10 MB per image
   +  Multiple image upload supported

4. **View Results**:
   - Original image
   - Enhanced image
   - Lane detection overlay (cyan lines)
   - Metrics: PSNR, Entropy, Processing time, Detection confidence

+5. **Batch Processing**:
+   - Multiple images can be uploaded simultaneously.
+   - Images are processed sequentially using the same parameters.
+   - Suitable for testing multiple low-light datasets.

## 📊 API Endpoints

### POST `/process`
Process image with enhancement and lane detection.

**Request**:
```json
{
  "image": <file>,
  "eta": 45,
  "lambda_val": 7.0,
  "baseline": "enhanced"
}
```

**Response**:
```json
{
  "status": "success",
  "original_img": "data:image/png;base64,...",
  "enhanced_img": "data:image/png;base64,...",
  "lane_detection_img": "data:image/png;base64,...",
  "metrics": {
    "psnr": 18.45,
    "entropy_original": 5.234,
    "entropy_enhanced": 6.891,
    "processing_time": 945.32,
    "detection_confidence": 78.5
  },
  "histogram_data": {
    "original": [...],
    "enhanced": [...]
  }
}
```

## Usage Guide

### Landing Page (/)
- View information about LaneVision's capabilities.
- Explore the processing pipeline.
- Click "Open Control Panel" to get started.

### Control Panel (/control)

1. **Upload Image**:
   - Drag & drop an image into the upload zone.
   - Alternatively, click "Browse Files" to select an image.
   - Supported formats: PNG, JPG, JPEG, TIFF.

2. **Enhancement Settings**:
   - **Contrast Factor**: Adjust the brightness enhancement level (1-100), default is 25.
   - **Noise Reduction**: Set the noise filter strength (0-10), default is 4.0.
   - **Baseline Model**: Select the model for comparison.

3. **Processing**:
   - Click "Process Image".
   - Wait for the processing to complete (indicated by the loading overlay).
   - View the side-by-side comparison of the original and enhanced images.

4. **Results**:
   - Detection confidence and processing time are displayed below the images.

## API Endpoints

### POST `/api/enhance`
Endpoint for image enhancement.

**Parameters (multipart/form-data)**:
- `image`: The image file to process.
- `contrast_h`: Contrast factor (1-100).
- `noise_a`: Noise reduction strength (0-10).

**Response**:
```json
{
    "success": true,
    "message": "Image berhasil diproses",
    "output_image": "/uploads/enhanced_image.jpg",
    "confidence": 98,
    "processing_time": "<15ms"
}
```

## Algorithm Details

### Improved Retinex Model
- Multi-scale Retinex for luminance recovery.
- Guided filtering for edge preservation.
- Contrast adjustment using hyperbolic tangent mapping.

### Lane Detection
- Canny edge detection.
- Hough line transform for lane boundary detection.
- Confidence score calculation based on edge strength.

## Requirements

- Python 3.8+
- Flask 3.0.0
- OpenCV 4.8.1.78
- NumPy 1.24.3

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Notes

- Maximum file upload size: 16 MB.
- Processing time may vary depending on the image resolution.
- All uploaded images are temporarily stored in the `uploads/` directory.

  ### Docker Notes

+- Docker ensures all dependencies run consistently.
+- Recommended for deployment and collaborative development.
+
+### Uploads Directory
+
+- Processed images may be temporarily stored in the `uploads/` folder.
+- The folder is automatically created if it does not exist.

+## Performance Optimization
+
+- Image processing results can be cached for faster repeated access.
+- Docker containers isolate dependencies and improve portability.
+- Batch processing enables efficient dataset evaluation.
## Troubleshooting

### Port 5000 is already in use
Edit `app.py` and change the port at the bottom:
```python
app.run(debug=True, port=5001)  # Change to a different port
```

### Unsupported image format
Ensure your image is in one of the supported formats: PNG, JPG, JPEG, or TIFF.

### OpenCV installation issues
If you encounter errors related to OpenCV, try upgrading it:
```bash
pip install --upgrade opencv-python
```

## License

This project is created for educational purposes as part of the Digital Image Processing (Pengolahan Citra Digital) course.
