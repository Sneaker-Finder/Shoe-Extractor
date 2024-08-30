import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def process(train_dir, n_clusters=20):

    images_dir = os.path.join(train_dir, "images")

    # Go through all image files
    for image_file in os.listdir(images_dir):

        image_path = os.path.join(images_dir, image_file)

        try:
            # Load image
            image = np.array(Image.open(image_path).convert('RGB'))
            height, width, _ = image.shape

            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            kmeans.fit(image.reshape(-1, 3))
            clustered_image = kmeans.cluster_centers_[kmeans.labels_].reshape(height, width, 3).astype(np.uint8)

            # Save the new image
            new_image = Image.fromarray(clustered_image)
            new_image.save(image_path)

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")