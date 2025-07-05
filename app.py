import cv2
import numpy as np
import base64
import traceback
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def is_valid_satellite_image(image):
    """
    Validate whether an image is a satellite image based on:
    1. Edge complexity (to avoid smooth images like faces).
    2. Laplacian variance (to check sharpness).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection: Canny algorithm
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / (image.shape[0] * image.shape[1])  # Edge density

    # Blur detection: Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # **New Validation Criteria**  
    # - Edge density must be high (satellite images have high details)
    # - Laplacian variance must be high (to ensure the image is not blurred)
    return edge_ratio > 0.05 and laplacian_var > 100  # Adjustable thresholds

def process_lulc(image):
    """
    Process the given satellite image to classify land cover into:
    - Houses (Red)
    - Natural Cover (Green) [Trees + Water]
    - Remaining Land (Yellow)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges
    house_lower = np.array([0, 0, 100], dtype=np.uint8)
    house_upper = np.array([50, 50, 255], dtype=np.uint8)

    natural_lower1 = np.array([30, 40, 40], dtype=np.uint8)  # Trees (Normal green)
    natural_upper1 = np.array([90, 255, 255], dtype=np.uint8)

    natural_lower2 = np.array([80, 40, 40], dtype=np.uint8)  # Water (Dark blue)
    natural_upper2 = np.array([140, 255, 255], dtype=np.uint8)

    # Create masks
    house_mask = cv2.inRange(hsv, house_lower, house_upper)
    natural_mask1 = cv2.inRange(hsv, natural_lower1, natural_upper1)
    natural_mask2 = cv2.inRange(hsv, natural_lower2, natural_upper2)
    natural_mask = cv2.bitwise_or(natural_mask1, natural_mask2)  # Combine both

    # Count Pixels
    total_pixels = image.shape[0] * image.shape[1]
    house_pixels = np.count_nonzero(house_mask)
    natural_pixels = np.count_nonzero(natural_mask)
    remaining_pixels = total_pixels - (house_pixels + natural_pixels)

    # Calculate percentages
    house_percentage = (house_pixels / total_pixels) * 100
    natural_percentage = (natural_pixels / total_pixels) * 100
    remaining_percentage = (remaining_pixels / total_pixels) * 100

    # Apply color overlay
    lulc_map = np.zeros_like(image)
    lulc_map[house_mask > 0] = [0, 0, 255]  # Red (Houses)
    lulc_map[natural_mask > 0] = [0, 255, 0]  # Green (Natural Cover: Trees + Water)
    lulc_map[(house_mask == 0) & (natural_mask == 0)] = [0, 255, 255]  # Yellow (Land)

    # Convert processed image to base64
    _, buffer = cv2.imencode('.png', lulc_map)
    lulc_map_base64 = base64.b64encode(buffer).decode('utf-8')

    return lulc_map_base64, house_percentage, natural_percentage, remaining_percentage

@app.route("/", methods=["GET"])
def home():
    return render_template("lulc.html")

@app.route("/process", methods=["POST"])
def process_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        image_np = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        # Validate if it's a satellite image
        if not is_valid_satellite_image(image):
            return jsonify({"error": "Uploaded image is not a valid satellite image"}), 400

        # Process the image
        lulc_map, house_pct, natural_pct, remaining_pct = process_lulc(image)

        # Convert original image to base64
        _, buffer = cv2.imencode(".png", image)
        original_image_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "original_image": original_image_base64,
            "lulc_map": lulc_map,
            "house_percentage": round(house_pct, 2),
            "natural_cover_percentage": round(natural_pct, 2),  # Combined Trees + Water
            "remaining_percentage": round(remaining_pct, 2)
        })

    except Exception as e:
        print(traceback.format_exc())  # Print full error in console
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
