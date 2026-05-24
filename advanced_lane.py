import cv2
import numpy as np

def detect_curved_lanes(image: np.ndarray) -> tuple:
    """
    Advanced Lane Finding menggunakan Bird's Eye View & Sliding Window Polynomial Fit.
    Mampu mendeteksi dan menggambar jalan yang melengkung (berbelok).
    """
    height, width = image.shape[:2]
    
    # ============================================================
    # STEP 1: Perspective Transform (Bird's Eye View)
    # ============================================================
    # Tentukan area trapesium jalan di depan kamera (dibuat FULL layar bawah)
    src = np.float32([
        [width * 0.30, height * 0.45],   # Top left (lebih lebar di ujung jalan)
        [width * 0.70, height * 0.45],   # Top right (lebih lebar di ujung jalan)
        [width * 1.0, height],           # Bottom right (mentok pinggir kanan)
        [width * 0.0, height]            # Bottom left (mentok pinggir kiri)
    ])
    
    # Area tujuan (dibentangkan menjadi persegi panjang dari atas)
    dst = np.float32([
        [width * 0.2, 0],
        [width * 0.8, 0],
        [width * 0.8, height],
        [width * 0.2, height]
    ])
    
    # Matriks transformasi
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src) # Untuk mengembalikan gambar nanti
    
    # Bengkokkan gambar menjadi tampak atas
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    
    # ============================================================
    # STEP 2: Thresholding (Mencari area terang/jalan)
    # ============================================================
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Peningkatan kontras
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Thresholding untuk mengambil piksel yang cukup terang
    _, binary_warped = cv2.threshold(equalized, 120, 255, cv2.THRESH_BINARY)
    
    # ============================================================
    # STEP 3: Sliding Window untuk melacak kurva jalan
    # ============================================================
    # Ambil histogram dari separuh bawah gambar untuk mencari pangkal lajur
    histogram = np.sum(binary_warped[height//2:, :], axis=0)
    
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Fallback jika tidak terdeteksi base yang jelas
    if leftx_base < 50: leftx_base = int(width * 0.2)
    if rightx_base > width - 50: rightx_base = int(width * 0.8)
    
    nwindows = 9
    window_height = int(height // nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 80
    minpix = 40
    
    left_lane_inds = []
    right_lane_inds = []
    
    # Melakukan sliding window dari bawah ke atas
    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass
        
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # ============================================================
    # STEP 4: Fit Polinomial (Parabola) untuk Garis Melengkung
    # ============================================================
    ploty = np.linspace(0, height - 1, height)
    confidence = 80.0
    
    if len(leftx) > 100 and len(rightx) > 100:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        left_fitx_raw = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx_raw = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Cari garis tengah lengkungan dari pantulan lampu (ini akan meliuk mengikuti jalan!)
        center_fitx = (left_fitx_raw + right_fitx_raw) / 2.0
        
        # Ekpansi (lebarkan) garis tengah tersebut ke pinggir kiri dan kanan
        # agar membentuk jalan yang sangat lebar tapi tetap melengkung!
        left_fitx = center_fitx - (width * 0.45)
        right_fitx = center_fitx + (width * 0.45)
        
    else:
        # Fallback lurus ke kiri/kanan mentok agar FULL JALAN jika gelap total
        left_fitx = np.ones_like(ploty) * 0
        right_fitx = np.ones_like(ploty) * width
        
    # ============================================================
    # STEP 5: Kembalikan proyeksi ke gambar asli (Unwarp)
    # ============================================================
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Format koordinat untuk cv2.fillPoly
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Gambar area hijau di antara lajur
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Gambar garis lajur itu sendiri
    cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 255, 0), thickness=25)   # Cyan
    cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 255, 255), thickness=25)  # Yellow
    
    # Transformasi balik (Unwarp) dari Bird's Eye View ke Perspektif Kamera Normal
    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    
    # Gabungkan dengan gambar asli
    result = cv2.addWeighted(image, 1, newwarp, 0.4, 0)
    
    # Menghasilkan citra debugging
    debug_img = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)
    
    return result, confidence, debug_img, newwarp
