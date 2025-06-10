import cv2

# Load the image
image = cv2.imread('./img/image.jpg')

#Display the original image
# cv2.imshow('Original Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Save the grayscale image
cv2.imwrite('./img/gray_image.jpg', gray_image)

# Display the grayscale image
# cv2.imshow('Grayscale Image', gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Delete the grayscale image file
import os
if os.path.exists('./img/gray_image.jpg'):
    os.remove('./img/gray_image.jpg')

# Resize to fixed dimensions
resized_image = cv2.resize(image, (200, 200))

# Keep the aspect ratio
height, width = image.shape[:2]
new_width = 300
new_height = int((new_width / width) * height)
resized_aspect_image = cv2.resize(image, (new_width, new_height))


# Display the resized images
# cv2.imshow('Resized Image', resized_image)
# cv2.imshow('Resized Aspect Image', resized_aspect_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Normalize the image naively
normalized_image = resized_aspect_image / 255.0

# Normalize the image using OpenCV
normalized_cv_image = cv2.normalize(resized_aspect_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Display the normalized images
cv2.imshow('Normalized Image (Naive)', normalized_image)
cv2.imshow('Normalized Image (OpenCV)', normalized_cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()