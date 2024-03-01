**Overview**

This 2D Object Recognition System is designed to accurately identify and classify objects placed on a uniform background in images. Leveraging advanced computer vision techniques and algorithms, the system offers high precision in distinguishing between various objects regardless of their orientation, scale, or position within the image. 

**Features**

- Threshold the input image/Video: implement a thresholding algorithm to separate objects from the background and generate binary image

- Clean up the binary image: use morphological filtering to address holes in the threshold image
- Segment the image into regions: Perform connected components analysis to identify regions
  - Display: Enhance region maps with a color palette for clarity
- Compute features for each major region: calculate features like the axis of least central mooment, oriented bounding box of identified regions and hu moments
- Collect training data: implement a training mode to collect and label feature vectors and storing them in a csv file
- Classify New Images: Use the object database to classify new objects, employing a scaled Euclidean distance metrix
  - Labeling: indicate the label of recognized objects in the output stream
- Implement a second classification method: implement a K-Nearest Neighbor (KNN) classifier
