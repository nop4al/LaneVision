from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_caching import Cache
import numpy as np
import cv2
import base64
import time
import os
from io import BytesIO

from algorithm import enhance_image, detect_lane_lines, calculate_psnr, calculate_entropy
from advanced_lane import detect_curved_lanes

app = Flask(__name__)
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})
CORS(app)

# ── Configuration ────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Create uploads directory (kept for potential future use, not used for processing)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image: np.ndarray) -> str:
    """Encode an OpenCV image (BGR or grayscale) to a base64 PNG data-URI string."""
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


def compute_histogram(image: np.ndarray) -> list:
    """Compute a 256-bin grayscale histogram and return as a list of ints."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten().astype(int).tolist()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Landing page for LaneVision."""
    return render_template('index.html')


@app.route('/control')
def control():
    """Control panel for image enhancement and lane detection."""
    return render_template('control.html')


@app.route('/process', methods=['POST'])
def process():
    """
    Process an uploaded image: enhance it, detect lane lines, and return
    the results with quality metrics and histogram data — all in memory.
    """
    try:
        # ── 1. Validate file presence ────────────────────────────────────
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image file provided in the request."
            }), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "status": "error",
                "message": "Unsupported file format. Please use PNG, JPG, JPEG, or TIFF."
            }), 400

        # ── 2. Read parameters ───────────────────────────────────────────
        eta = float(request.form.get('eta', 25))
        lambda_val = float(request.form.get('lambda_val', 4.0))
        baseline = request.form.get('baseline', 'standard')

        if baseline not in ('standard', 'enhanced'):
            baseline = 'standard'

        # ── 3. Load image from memory ────────────────────────────────────
        file_bytes = file.read()
        np_array = np.frombuffer(file_bytes, np.uint8)
        original_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if original_image is None:
            return jsonify({
                "status": "error",
                "message": "Failed to decode image. The file may be corrupted."
            }), 400
            
        cache_key = f"{file.filename}_{eta}_{lambda_val}_{baseline}"
        cached_result = cache.get(cache_key)
             
        if cached_result:
             return jsonify(cached_result)

        # ── 4. Process ───────────────────────────────────────────────────
        start_time = time.time()

        # Enhancement with baseline model selection
        enhanced_image = enhance_image(original_image, eta, lambda_val, baseline)

        # Lane detection menggunakan Advanced Curved Lane Finding
        lane_detection_image, detection_confidence, canny_edges_img, roi_masked_img = detect_curved_lanes(enhanced_image)

        # Metrics
        psnr_value = calculate_psnr(original_image, enhanced_image)
        entropy_original = calculate_entropy(original_image)
        entropy_enhanced = calculate_entropy(enhanced_image)

        processing_time_ms = (time.time() - start_time) * 1000

        # ── 5. Encode results ────────────────────────────────────────────
        original_b64 = image_to_base64(original_image)
        enhanced_b64 = image_to_base64(enhanced_image)
        lane_b64 = image_to_base64(lane_detection_image)
        canny_b64 = image_to_base64(canny_edges_img)
        roi_b64 = image_to_base64(roi_masked_img)
        
        timestamp = int(time.time())
        
        enhanced_filename = f"enhanced_{timestamp}.png"
        lane_filename = f"lane_{timestamp}.png"
        
        enhanced_path = os.path.join(UPLOAD_FOLDER, enhanced_filename)
        lane_path = os.path.join(UPLOAD_FOLDER, lane_filename)

        cv2.imwrite(enhanced_path, enhanced_image)
        cv2.imwrite(lane_path, lane_detection_image)
        
        # Histograms
        hist_original = compute_histogram(original_image)
        hist_enhanced = compute_histogram(enhanced_image)

        # ── 6. Build response ────────────────────────────────────────────
        response_data= {
            "status": "success",
            "original_img": original_b64,
            "enhanced_img": enhanced_b64,
            "lane_detection_img": lane_b64,
            "canny_edges_img": canny_b64,
            "roi_masked_img": roi_b64,
            "download_files":{
                "enhanced":f"/static/uploads/{enhanced_filename}",
                "lane_detection":f"/static/uploads/{lane_filename}"
            },
            "metrics": {
                "psnr": round(psnr_value, 2),
                "entropy_original": round(entropy_original, 3),
                "entropy_enhanced": round(entropy_enhanced, 3),
                "processing_time": round(processing_time_ms, 2),
                "detection_confidence": round(detection_confidence, 1)
            },
            "histogram_data": {
                "original": hist_original,
                "enhanced": hist_enhanced
            }
        }
        cache.set(cache_key, response_data)
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Processing failed: {str(e)}",
            "error_type": type(e).__name__,
            "timestamp": time.time()
        }), 400


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
