import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

class EOAPIPreprocessor:
    """
    Preprocessor for EO API Sentinel-1 images.
    Converts data from linear power to dB scale.
    """

    def __init__(self, eoapi_path, output_path):
        """
        Initialize the preprocessor.
        
        Parameters:
        - eoapi_path: str, path to the EO API original GeoTIFF
        - output_path: str, path to save the converted GeoTIFF
        """
        self.eoapi_path = eoapi_path
        self.output_path = output_path

    def convert_to_db(self):
        """
        Convert EO API image from linear power to dB scale.
        Saves the output to self.output_path.
        """
        with rasterio.open(self.eoapi_path) as src:
            profile = src.profile
            data = src.read(1).astype('float32')  # read first band

        # Avoid log(0) issues by replacing non-positive values
        data[data <= 0] = np.nanmin(data[data > 0])  

        # Convert to dB
        data_db = 10 * np.log10(data)

        # Update profile for output
        profile.update(dtype='float32', nodata=np.nan)

        # Save converted image

        with rasterio.open(self.output_path, 'w', **profile) as dst:
            dst.write(data_db, 1)

        print(f"✅ EO API image converted to dB and saved as {self.output_path}")
        return data_db

    def visualize_conversion(self):
        """
        Visualize EO API before and after conversion to dB.
        """
        with rasterio.open(self.eoapi_path) as src:
            original = src.read(1)

        with rasterio.open(self.output_path) as src:
            converted = src.read(1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original EO API (linear power)
        im1 = axes[0].imshow(original, cmap="gray")
        axes[0].set_title("EO API (Linear Power)")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Converted EO API (dB)
        im2 = axes[1].imshow(converted, cmap="gray", vmin=-35, vmax=-15)
        axes[1].set_title("EO API (Converted to dB)")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        plt.show()
