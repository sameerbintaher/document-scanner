import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def main():
    # Load the image
    image = cv2.imread('docs.jpg')
    if image is None:
        print("Error: Could not read the image file.")
        return
    cv2.imwrite('01_original.jpg', image)
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('02_grayscale.jpg', gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite('03_blurred.jpg', blurred)

    # Edge detection
    edged = cv2.Canny(blurred, 30, 100)  # Adjusted thresholds
    cv2.imwrite('04_edged.jpg', edged)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Increased to top 10 contours

    # Visualize all contours
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite('05_all_contours.jpg', contour_image)

    # Find the document contour
    doc_contour = None
    for i, c in enumerate(contours):
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            doc_contour = approx
            print(f"Document contour found. It was contour number {i+1}")
            break

    if doc_contour is None:
        print("No document contour found. Here are the details of the top 5 contours:")
        for i, c in enumerate(contours[:5]):
            print(f"Contour {i+1}: Area = {cv2.contourArea(c)}, Perimeter = {cv2.arcLength(c, True)}")
        return

    # Visualize the detected document contour
    cv2.drawContours(image, [doc_contour], -1, (0, 255, 0), 2)
    cv2.imwrite('06_detected_document.jpg', image)

    # Perspective transformation
    pts = doc_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    cv2.imwrite('07_warped.jpg', warped)

    # Convert to grayscale
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('08_warped_gray.jpg', warped_gray)

    # Apply different thresholding methods
    _, binary = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('09_binary_threshold.jpg', binary)

    adaptive = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite('10_adaptive_threshold.jpg', adaptive)

    print("All intermediate results have been saved. Please check the output images.")

if __name__ == "__main__":
    main()
