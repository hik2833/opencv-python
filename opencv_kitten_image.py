# Import OpenCV
import cv2

# Step 1: Read the uploaded kitten image
image_path = "shutterstock147979985--250.jpg"
image = cv2.imread(image_path)

# Extract each channel separately
blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]

# Display each channel as a 2D image
cv2.imshow('Blue Channel', blue_channel)
cv2.imshow('Green Channel', green_channel)
cv2.imshow('Red Channel', red_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Merge the channels back into one image (normal RGB)
merged_image = cv2.merge([blue_channel, green_channel, red_channel])

# Display merged image
cv2.imshow('Merged Image (BGR)', merged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Swap red and green channels to create a GRB image
swapped_image = cv2.merge([green_channel, red_channel, blue_channel])

# Display swapped image
cv2.imshow('Swapped Image (GRB)', swapped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
