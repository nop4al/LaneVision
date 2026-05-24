/**
 * control.js — Client-side logic for the LaneVision control panel.
 * Handles drag & drop, file preview, parameter sliders, and the
 * image-processing request to /process.
 */


document.addEventListener("DOMContentLoaded", function() {
    console.log("Control.js loaded successfully");

    // Get DOM elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const fileName = document.getElementById("fileName");
    const contrastSlider = document.getElementById("contrastSlider");
    const contrastValue = document.getElementById("contrastValue");
    const noiseSlider = document.getElementById("noiseSlider");
    const noiseValue = document.getElementById("noiseValue");
    const baselineSelect = document.getElementById("baselineSelect");
    const processBtn = document.getElementById("processBtn");
    const resultsSection = document.getElementById("resultsSection");
    const originalImage = document.getElementById("originalImage");
    const enhancedImage = document.getElementById("enhancedImage");
    const laneImage = document.getElementById("laneImage");
    const confidence = document.getElementById("confidence");
    const processingTime = document.getElementById("processingTime");
    const psnrValue = document.getElementById("psnrValue");
    const entropyEnhanced = document.getElementById("entropyEnhanced");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const errorBox = document.getElementById("errorBox");
    const histogramCanvas = document.getElementById("histogramChart");

    let currentFile = null;
    let chartInstance = null;

    // Show error message
    function showError(message) {
        console.error("Error:", message);
        if (errorBox) {
            errorBox.textContent = message;
            errorBox.style.display = "block";
            setTimeout(function() {
                errorBox.style.display = "none";
            }, 5000);
        } else {
            alert(message);
        }
    }

    // Hide error
    function hideError() {
        if (errorBox) {
            errorBox.style.display = "none";
        }
    }

    // Handle file selection
    function handleFile(file) {
        if (!file) return;
        
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff'];
        const maxSize = 10 * 1024 * 1024;
        
        if (!validTypes.includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|tiff)$/i)) {
            showError("Unsupported file format. Use PNG, JPG, JPEG, or TIFF.");
            return;
        }
        
        if (file.size > maxSize) {
            showError("File size exceeds 10 MB limit.");
            return;
        }
        
        currentFile = file;
        
        // Preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.style.display = "block";
            fileName.textContent = file.name;
        };
        reader.readAsDataURL(file);
        
        hideError();
    }

    // Drag & drop
    if (dropZone) {
        dropZone.addEventListener("dragover", function(e) {
            e.preventDefault();
            dropZone.style.borderColor = "#d4704e";
        });
        
        dropZone.addEventListener("dragleave", function(e) {
            e.preventDefault();
            dropZone.style.borderColor = "#2a2a32";
        });
        
        dropZone.addEventListener("drop", function(e) {
            e.preventDefault();
            dropZone.style.borderColor = "#2a2a32";
            const file = e.dataTransfer.files[0];
            if (file) {
                handleFile(file);
                if (fileInput) {
                    const dt = new DataTransfer();
                    dt.items.add(file);
                    fileInput.files = dt.files;
                }
            }
        });
    }
    
    // File input change
    if (fileInput) {
        fileInput.addEventListener("change", function() {
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });
    }
    
    // Sliders
    if (contrastSlider && contrastValue) {
        contrastSlider.addEventListener("input", function() {
            contrastValue.textContent = contrastSlider.value;
        });
    }
    
    if (noiseSlider && noiseValue) {
        noiseSlider.addEventListener("input", function() {
            noiseValue.textContent = parseFloat(noiseSlider.value).toFixed(1);
        });
    }
    
    // Render histogram using Canvas API (no external library needed)
    function renderHistogram(original, enhanced) {
        if (!histogramCanvas) return;
        
        const ctx = histogramCanvas.getContext("2d");
        const width = histogramCanvas.width;
        const height = histogramCanvas.height;
        
        // Clear canvas
        ctx.fillStyle = "#1a1a22";
        ctx.fillRect(0, 0, width, height);
        
        // Find max value for scaling
        const maxOriginal = Math.max(...original);
        const maxEnhanced = Math.max(...enhanced);
        const maxValue = Math.max(maxOriginal, maxEnhanced);
        
        if (maxValue === 0) return;
        
        const barWidth = width / 256;
        const padding = 40;
        const graphHeight = height - padding * 2;
        
        // Draw grid and labels
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = padding + (graphHeight / 5) * i;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }
        
        // Draw histograms
        for (let i = 0; i < 256; i++) {
            const x = padding + i * barWidth;
            
            // Original histogram (darker color)
            const origHeight = (original[i] / maxValue) * graphHeight;
            ctx.fillStyle = "rgba(212, 112, 78, 0.5)";
            ctx.fillRect(x, height - padding - origHeight, barWidth * 0.45, origHeight);
            
            // Enhanced histogram (brighter color)
            const enhHeight = (enhanced[i] / maxValue) * graphHeight;
            ctx.fillStyle = "rgba(46, 125, 50, 0.7)";
            ctx.fillRect(x + barWidth * 0.45, height - padding - enhHeight, barWidth * 0.45, enhHeight);
        }
        
        // Draw axes
        ctx.strokeStyle = "#666";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();
        
        // Labels
        ctx.fillStyle = "#a8a6a0";
        ctx.font = "12px Arial";
        ctx.textAlign = "center";
        
        // X-axis labels
        for (let i = 0; i <= 256; i += 64) {
            const x = padding + (i / 256) * (width - padding * 2);
            ctx.fillText(i, x, height - padding + 20);
        }
        
        // Y-axis label
        ctx.textAlign = "right";
        ctx.fillText("Frequency", padding - 10, 20);
        
        // Legend
        ctx.fillStyle = "rgba(212, 112, 78, 0.7)";
        ctx.fillRect(width - 180, 10, 15, 15);
        ctx.fillStyle = "#e8e6e1";
        ctx.textAlign = "left";
        ctx.font = "12px Arial";
        ctx.fillText("Original", width - 155, 22);
        
        ctx.fillStyle = "rgba(46, 125, 50, 0.7)";
        ctx.fillRect(width - 180, 30, 15, 15);
        ctx.fillStyle = "#e8e6e1";
        ctx.fillText("Enhanced", width - 155, 42);
    }
    
    // Process image
    window.enhanceImage = async function() {
        if (!currentFile && (!fileInput || !fileInput.files || fileInput.files.length === 0)) {
            showError("Please select an image first.");
            return;
        }
        
        const file = currentFile || fileInput.files[0];
        
        if (loadingOverlay) loadingOverlay.style.display = "flex";
        if (processBtn) processBtn.disabled = true;
        if (resultsSection) resultsSection.style.display = "none";
        hideError();
        
        const formData = new FormData();
        formData.append("image", file);
        formData.append("eta", contrastSlider ? contrastSlider.value : 45);
        formData.append("lambda_val", noiseSlider ? noiseSlider.value : 4);
        formData.append("baseline", baselineSelect ? baselineSelect.value : "standard");
        
        try {
            const response = await fetch("/process", {
                method: "POST",
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status === "success") {
                if (originalImage) originalImage.src = data.original_img;
                if (enhancedImage) enhancedImage.src = data.enhanced_img;
                if (laneImage) laneImage.src = data.lane_detection_img;
                if (confidence) confidence.textContent = data.metrics.detection_confidence + "%";
                if (processingTime) processingTime.textContent = data.metrics.processing_time + " ms";
                if (psnrValue) psnrValue.textContent = data.metrics.psnr;
                if (entropyEnhanced) entropyEnhanced.textContent = data.metrics.entropy_enhanced;
                
                if (data.histogram_data) {
                    renderHistogram(data.histogram_data.original, data.histogram_data.enhanced);
                }
                
                if (data.download_files) {
                    const downloadEnhanced = document.getElementById("downloadEnhanced");
                    const downloadLane = document.getElementById("downloadLane");
                    if (downloadEnhanced) downloadEnhanced.href = data.download_files.enhanced;
                    if (downloadLane) downloadLane.href = data.download_files.lane_detection;
                }
                
                if (resultsSection) {
                    resultsSection.style.display = "block";
                    resultsSection.scrollIntoView({ behavior: "smooth" });
                }
            } else {
                showError(data.message || "Processing failed.");
            }
        } catch (err) {
            console.error("Fetch error:", err);
            showError("Cannot connect to server. Make sure Flask is running on port 5000.");
        } finally {
            if (loadingOverlay) loadingOverlay.style.display = "none";
            if (processBtn) processBtn.disabled = false;
        }
    };
    
    console.log("Control.js initialization complete");
});
