import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2 

image = sys.argv[1]

bgr = cv2.imread(image , cv2.IMREAD_COLOR)  # uint8 image
bgrNormalised = bgr/255.0

# Converting image to RGB 
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgbNormalised = rgb/255.0

##MatplotLib Images
fig = plt.figure()
fig.canvas.set_window_title("Matplotlib Images")
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(rgb)
ax1.title.set_text("Original Image")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(rgbNormalised)
ax2.title.set_text("Normalized Image")
plt.show()

# OpenCV Images
cv2.imshow('Original Image',bgr)
cv2.imshow('Normalized Image',bgrNormalised)
cv2.waitKey(0) ### Destroy all windows when any key is pressed
cv2.destroyAllWindows()






