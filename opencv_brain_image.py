
# Import OpenCV
import cv2
# Step 1: Read the uploaded image
image_path = "shutterstock93075775--250 2.jpg"
image = cv2.imread(image_path)
# Step 2: Display the image
cv2.imshow('Brain Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Step 3: Save a copy of the image to a new location
output_path = "brain_image_copy.jpg"  # This will save in the same directory as your script
cv2.imwrite(output_path, image)
