import cv2
import numpy as np

# List to store clicked points
points = []

def click_event(event, x, y, flags, params):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        # Draw the point on the image
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", img)

        if len(points) == 2:
            # Draw a line between the points
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow("Image", img)

            # Calculate Euclidean distance
            dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
            print(f"Distance between points: {dist:.2f} pixels")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Load image
img = cv2.imread("BEV.png")  # Replace with your image path
if img is None:
    raise FileNotFoundError("Image file not found.")

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
