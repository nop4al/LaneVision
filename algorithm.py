"""
algorithm.py — Low-light image enhancement for night-time lane detection
Based on: "Image Adaptive Contrast Enhancement for Low-illumination Lane Lines
Based on Improved Retinex and Guided Filter" (Hui Ma et al., 2021)
"""

import numpy as np
import cv2
import math


# ── 1. Luminance Channel Estimation ─────────────────────────────────────────

def estimate_luminance_channel(image: np.ndarray) -> np.ndarray:
    """
    Estimate the optimal luminance channel Lw by searching for the
    weight combination (wr, wg, wb) that minimises the bimodal energy
    function (Equation 6).  Weights are discretised with step 0.1.
    """
    # Ensure image is uint8 for cvtColor
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        # cvtColor must work on uint8, then convert to float64
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    else:
        img_rgb = image.astype(np.float64) / 255.0

    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    delta = 1e-6
    sigma = 0.5

    def gradient_magnitude(channel):
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)

    grad_R = gradient_magnitude(R)
    grad_G = gradient_magnitude(G)
    grad_B = gradient_magnitude(B)

    best_energy = np.inf
    best_weights = (1.0/3.0, 1.0/3.0, 1.0/3.0)

    def normal_pdf(x, sigma):
        return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)

    steps = np.arange(0.0, 1.01, 0.1)
    for wr in steps:
        for wg in steps:
            wb = 1.0 - wr - wg
            if wb < -1e-9 or wb > 1.0 + 1e-9:
                continue
            wb = np.clip(wb, 0.0, 1.0)

            L_grad = wr * grad_R + wg * grad_G + wb * grad_B
            energy = normal_pdf(L_grad + delta, sigma) + normal_pdf(L_grad - delta, sigma)
            E = -np.sum(np.log(energy + 1e-30))

            if E < best_energy:
                best_energy = E
                best_weights = (wr, wg, wb)

    wr, wg, wb = best_weights
    Lw = wr * R + wg * G + wb * B
    Lw = np.clip(Lw, 0.0, 1.0)

    return Lw


# ── 2. Global Tone Mapping ──────────────────────────────────────────────────

def global_tone_mapping(Lw: np.ndarray) -> np.ndarray:
    """Apply global tone mapping to compress dynamic range."""
    delta = 1e-6
    L_bar = np.exp(np.mean(np.log(delta + Lw)))
    Lw_max = np.max(Lw)
    if Lw_max < delta:
        return np.zeros_like(Lw)
    numerator = np.log(Lw / L_bar + 1.0)
    denominator = np.log(Lw_max / L_bar + 1.0)
    if abs(denominator) < 1e-10:
        return np.zeros_like(Lw)
    Lg = numerator / denominator
    return np.clip(Lg, 0.0, 1.0)


# ── 3. Guided Filter ────────────────────────────────────────────────────────

def guided_filter(guide: np.ndarray, src: np.ndarray,
                  radius: int = 10, eps: float = 0.01) -> np.ndarray:
    """Manual implementation of guided filter."""
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide, ddepth=-1, ksize=ksize)
    mean_p = cv2.boxFilter(src, ddepth=-1, ksize=ksize)
    mean_Ip = cv2.boxFilter(guide * src, ddepth=-1, ksize=ksize)
    mean_II = cv2.boxFilter(guide * guide, ddepth=-1, ksize=ksize)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize)
    return mean_a * guide + mean_b


# ── 4. Hyperbolic Tangent Mapping ────────────────────────────────────────────

def hyperbolic_tangent_mapping(Lg: np.ndarray, guided_output: np.ndarray,
                               eta: float = 25, lambda_val: float = 4) -> np.ndarray:
    """Apply adaptive hyperbolic tangent mapping."""
    delta = 1e-6
    Lg_max = np.max(Lg)
    if Lg_max < delta:
        Lg_max = delta

    Ga = 1.0 + eta * (Lg / Lg_max)
    L_bar_g = np.exp(np.mean(np.log(delta + Lg)))
    Oc = lambda_val * L_bar_g
    ratio = Lg / (guided_output + delta)
    R_final = Ga * np.tanh(ratio + Oc)

    R_max = R_final.max()
    if R_max > 1e-8:
        R_final = R_final / R_max
    return np.clip(R_final, 0.0, 1.0)


# ── 5. Enhancement Pipelines ────────────────────────────────────────────────

def enhanced_retinex_pipeline(image: np.ndarray, eta: float = 45,
                              lambda_val: float = 4.0) -> np.ndarray:
    """Enhanced Retinex pipeline untuk kondisi malam hari (agresif)."""
    try:
        work = image.copy()

        # Gamma correction ekstrim untuk malam hari
        gamma = 1.0 / (1.0 + eta * 0.12)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        brightened = cv2.LUT(work, lut)

        # Multiple CLAHE passes
        lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe1 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
        l = clahe1.apply(l)
        clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe2.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        brightened = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Denoise
        denoise_h = max(8, int(8 + lambda_val * 3.0))
        denoised = cv2.fastNlMeansDenoisingColored(brightened, None, denoise_h, denoise_h, 15, 21)

        # Retinex pipeline
        Lw = estimate_luminance_channel(denoised)
        Lw = cv2.GaussianBlur(Lw, (11, 11), 0)
        Lg = global_tone_mapping(Lw)
        guided_output = guided_filter(Lg, Lg, radius=30, eps=0.02)
        R_final = hyperbolic_tangent_mapping(Lg, guided_output, eta * 1.5, lambda_val)

        # Reconstruct image
        lab_out = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB).astype(np.float64)
        lab_out[:, :, 0] = np.clip(R_final * 255.0, 0, 255)
        lab_out[:, :, 1] = cv2.GaussianBlur(lab_out[:, :, 1], (15, 15), 0)
        lab_out[:, :, 2] = cv2.GaussianBlur(lab_out[:, :, 2], (15, 15), 0)
        enhanced = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Final contrast boost
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=15)

        return enhanced
    except Exception as e:
        print(f"[Enhanced Retinex] error: {e}")
        return image.copy()


def standard_yolov8_pipeline(image: np.ndarray, eta: float = 45,
                            lambda_val: float = 4.0) -> np.ndarray:
    """Standard pipeline untuk kondisi malam hari."""
    try:
        work = image.copy()

        # Gamma correction
        gamma = 1.0 / (1.0 + eta * 0.05)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        brightened = cv2.LUT(work, lut)

        # CLAHE
        lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        brightened = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Denoise
        denoise_h = max(5, int(5 + lambda_val * 2.0))
        denoised = cv2.fastNlMeansDenoisingColored(brightened, None, denoise_h, denoise_h, 10, 21)

        # Luminance pipeline
        Lw = estimate_luminance_channel(denoised)
        Lw = cv2.GaussianBlur(Lw, (7, 7), 0)
        Lg = global_tone_mapping(Lw)
        guided_output = guided_filter(Lg, Lg, radius=15, eps=0.04)
        R_final = hyperbolic_tangent_mapping(Lg, guided_output, eta * 1.0, lambda_val)

        # Reconstruct
        lab_out = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB).astype(np.float64)
        lab_out[:, :, 0] = np.clip(R_final * 255.0, 0, 255)
        enhanced = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return enhanced
    except Exception as e:
        print(f"[Standard YOLOv8] error: {e}")
        return image.copy()


def enhance_image(image: np.ndarray, eta: float = 45,
                  lambda_val: float = 4.0, baseline: str = 'standard') -> np.ndarray:
    """End-to-end low-light enhancement pipeline."""
    if baseline == 'enhanced':
        return enhanced_retinex_pipeline(image, eta, lambda_val)
    else:
        return standard_yolov8_pipeline(image, eta, lambda_val)


# ── 6. Lane-Line Detection (OPTIMIZED FOR NIGHT) ────────────────────────────

def detect_lane_lines(image: np.ndarray) -> tuple:
    """
    Detect lane lines - OPTIMIZED UNTUK MALAM HARI.
    Menggunakan Bilateral Filter, Canny edges + Median Slope-filtered Hough.
    Mengembalikan (lanes_img, confidence, canny_edges, roi_masked).
    """
    height, width = image.shape[:2]
    
    # ============================================================
    # STEP 1: Convert to grayscale & preprocess
    # ============================================================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gamma correction untuk mencerahkan gambar gelap
    gamma = 0.4
    lookup = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    brightened = cv2.LUT(gray, lookup)
    
    # CLAHE untuk kontras lokal
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    equalized = clahe.apply(brightened)
    
    # Bilateral Filter: Meredam grain noise, menjaga tepi jalan
    blurred = cv2.bilateralFilter(equalized, d=5, sigmaColor=50, sigmaSpace=50)
    
    # ============================================================
    # STEP 2: Canny Edge Detection
    # ============================================================
    # Menggunakan threshold yang lebih rendah/sensitif untuk menangkap tepi jalan yang pudar
    edges = cv2.Canny(blurred, 15, 50)
    
    # ============================================================
    # STEP 3: Morphological Operations
    # ============================================================
    kernel_dilate = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel_dilate, iterations=1)
    
    # ============================================================
    # STEP 4: Region of Interest (Fokus ke area jalan bawah)
    # ============================================================
    # Mempersempit ROI agar rumah dan pohon tidak masuk
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (int(width * 0.15), height),                      
        (int(width * 0.40), int(height * 0.55)),          
        (int(width * 0.60), int(height * 0.55)),          
        (int(width * 0.85), height)                       
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_vertices, 255)
    
    # Sangat Penting: Abaikan area watermark 'detikcom' di kanan bawah!
    cv2.rectangle(mask, (int(width * 0.7), int(height * 0.85)), (width, height), 0, -1)
    
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # ============================================================
    # STEP 5: Hough Line Detection
    # ============================================================
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=15,           
        minLineLength=10,       
        maxLineGap=50          
    )
    
    # ============================================================
    # STEP 6: Line Filtering & Drawing (Median Slope Fitting)
    # ============================================================
    lanes_img = image.copy()
    confidence = 0.0
    
    if lines is not None and len(lines) > 0:
        left_lines = []
        right_lines = []
        
        # Titik Hilang (Vanishing Point) diasumsikan berada di tengah ufuk jalan
        vp_x = width // 2
        vp_y = int(height * 0.55)
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0:
                continue 
                
            slope = dy / dx
            length = np.sqrt(dx**2 + dy**2)
            
            color = (0, 0, 255) # Red (Ditolak)
            thickness = 1
            
            if length > 10:
                x_center = (x1 + x2) / 2.0
                
                # Filter Kemiringan
                if -3.0 < slope < -0.3 and x_center < width * 0.55:
                    left_lines.append([x1, y1, x2, y2, slope])
                    color = (0, 255, 0) # Green (Diterima)
                    thickness = 2
                elif 0.3 < slope < 3.0 and x_center > width * 0.45:
                    right_lines.append([x1, y1, x2, y2, slope])
                    color = (0, 255, 0) # Green (Diterima)
                    thickness = 2
                    
            # [FITUR DEBUG DIMATIKAN UNTUK HASIL FINAL]
            # cv2.line(lanes_img, (x1, y1), (x2, y2), color, thickness)
        
        # ── Fit Lajur Kiri dengan Vanishing Point Anchoring & Prior ──
        # Memaksa garis melewati titik hilang agar membentuk perspektif 3D yang benar (tidak akan pernah menyilang X)
        if len(left_lines) > 0:
            cx = [(pts[0] + pts[2]) / 2.0 for pts in left_lines]
            cy = [(pts[1] + pts[3]) / 2.0 for pts in left_lines]
            cx_med = np.median(cx)
            cy_med = np.median(cy)
            
            if cy_med > vp_y + 5: # Pastikan titik noise berada di bawah titik hilang
                # Lane Width Prior: Jika titik deteksi terlalu di tengah (kemungkinan besar pantulan lampu)
                if vp_x - cx_med < width * 0.15:
                    m_vp = -1.2 # Kemiringan prior standar lajur kiri
                else:
                    m_vp = (vp_y - cy_med) / (vp_x - cx_med + 1e-6)
                
                y1_fit = height
                y2_fit = vp_y
                x1_fit = int(vp_x + (y1_fit - vp_y) / m_vp)
                x2_fit = vp_x
                
                cv2.line(lanes_img, (x1_fit, y1_fit), (x2_fit, y2_fit), (255, 255, 0), 5)  # Cyan
        else:
            # Fallback prior jika garis tepi aspal sama sekali tidak terlihat (gelap)
            y1_fit = height
            y2_fit = vp_y
            x1_fit = int(width * 0.15)
            x2_fit = vp_x
            cv2.line(lanes_img, (x1_fit, y1_fit), (x2_fit, y2_fit), (255, 255, 0), 5)
        
        # ── Fit Lajur Kanan dengan Vanishing Point Anchoring & Prior ──
        if len(right_lines) > 0:
            cx = [(pts[0] + pts[2]) / 2.0 for pts in right_lines]
            cy = [(pts[1] + pts[3]) / 2.0 for pts in right_lines]
            cx_med = np.median(cx)
            cy_med = np.median(cy)
            
            if cy_med > vp_y + 5:
                # Lane Width Prior untuk lajur kanan
                if cx_med - vp_x < width * 0.15:
                    m_vp = 1.2
                else:
                    m_vp = (vp_y - cy_med) / (vp_x - cx_med + 1e-6)
                
                y1_fit = height
                y2_fit = vp_y
                x1_fit = int(vp_x + (y1_fit - vp_y) / m_vp)
                x2_fit = vp_x
                
                cv2.line(lanes_img, (x1_fit, y1_fit), (x2_fit, y2_fit), (0, 255, 255), 5)  # Yellow
        else:
            # Fallback prior untuk lajur kanan
            y1_fit = height
            y2_fit = vp_y
            x1_fit = int(width * 0.85)
            x2_fit = vp_x
            cv2.line(lanes_img, (x1_fit, y1_fit), (x2_fit, y2_fit), (0, 255, 255), 5)
        
        total_valid = 0
        if len(left_lines) > 0: total_valid += 1
        if len(right_lines) > 0: total_valid += 1
        confidence = 60.0 + (total_valid * 20.0)
    else:
        confidence = 10.0
    
    return lanes_img, confidence, edges, masked_edges


# ── 7. Quality Metrics ──────────────────────────────────────────────────────

def calculate_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    original = original.astype(np.float64)
    enhanced = enhanced.astype(np.float64)
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_entropy(image: np.ndarray) -> float:
    """Compute Shannon entropy."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log2(p)))


def get_histogram_data(image: np.ndarray) -> list:
    """Compute 256-bin grayscale histogram."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten().astype(int).tolist()
