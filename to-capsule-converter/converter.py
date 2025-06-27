import cv2
import numpy as np

def convert_to_capsule_style_fisheye(input_path: str, output_path: str, output_size=(512, 512), zoom_factor=1.0):
    img = cv2.imread(input_path)
    h_orig, w_orig = img.shape[:2]

    img_resized = cv2.resize(img, output_size)

    h, w = output_size
    K = np.array([[w/2, 0, w/2],
                  [0, w/2, h/2],
                  [0,   0,   1]], dtype=np.float32)
    D = np.array([-0.4, 0.1, 0, 0], dtype=np.float32)

    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    fisheye_img = cv2.remap(img_resized, map1, map2, interpolation=cv2.INTER_LINEAR)

    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(center) - 5
    cv2.circle(mask, center, radius, 255, -1)

    result = np.zeros_like(fisheye_img)
    for c in range(3):
        result[:, :, c] = cv2.bitwise_and(fisheye_img[:, :, c], mask)

    if zoom_factor != 1.0:
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        zoomed = result[y1:y1 + new_h, x1:x1 + new_w]
        result = cv2.resize(zoomed, output_size)

    # Save output
    cv2.imwrite(output_path, result)
    print(f"Saved zoomed fisheye capsule-style image to {output_path}")

# Usage
convert_to_capsule_style_fisheye("./images/000001.png", "./converted-images/capsule_like_output.jpg", zoom_factor=1.13)

