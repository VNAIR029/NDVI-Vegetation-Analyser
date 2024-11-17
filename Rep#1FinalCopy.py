import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import seaborn as sns
from datetime import datetime
import glob


class NDVIAnalyzer:
    def __init__(self):
        self.vegetation_ranges = {
            'No Vegetation': (-1, 0),
            'Very Low': (0, 0.2),
            'Low': (0.2, 0.4),
            'Moderate': (0.4, 0.6),
            'High': (0.6, 0.8),
            'Very High': (0.8, 1)
        }

    def process_ndvi_image(self, image):
        """Process NDVI image and calculate index"""
        try:
            img_array = np.array(image)
            if len(img_array.shape) != 3:
                return None, "Error: Image must be RGB format"

            # Extract bands (assuming Landsat order: NIR = band 5, Red = band 4)
            nir = img_array[:, :, 1].astype(float)  # Using green channel for NIR
            red = img_array[:, :, 0].astype(float)  # Using red channel

            # Calculate NDVI
            ndvi = np.where(
                (nir + red) == 0,
                0,
                (nir - red) / (nir + red)
            )

            return ndvi, None
        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    def analyze_ndvi(self, ndvi_values):
        """Analyze NDVI values and generate assessment"""
        mean_ndvi = np.mean(ndvi_values)

        if mean_ndvi >= 0.5:
            assessment = "Low drought risk (Healthy vegetation)"
            status = "success"
        elif mean_ndvi >= 0.2:
            assessment = "Moderate drought risk (Watch conditions)"
            status = "warning"
        else:
            assessment = "High drought risk (Action needed)"
            status = "error"

        return mean_ndvi, assessment, status

    def generate_visualization(self, ndvi_values, plot_type="standard"):
        """Generate visualization plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "standard":
            sns.histplot(data=ndvi_values.flatten(), bins=50, ax=ax, color='green', alpha=0.6)
            ax.set_xlabel('NDVI Value')
            ax.set_ylabel('Frequency')
            ax.set_title('NDVI Distribution')
        elif plot_type == "heatmap":
            sns.heatmap(ndvi_values, cmap='RdYlGn', ax=ax)
            ax.set_title('NDVI Heatmap')

        return fig


def process_directory(directory_path):
    """Process all images in a directory"""
    analyzer = NDVIAnalyzer()
    results = []

    st.write(f"Looking for data in: {directory_path}")

    # Walk through landsat_collect directory
    for root, dirs, files in os.walk(directory_path):
        if "crop" in root:
            st.write(f"Processing directory: {root}")

            # Get the parent directory name
            dir_name = os.path.basename(os.path.dirname(root))
            st.write(f"Directory: {dir_name}")

            # Process images in directory
            image_files = glob.glob(os.path.join(root, "*.tif"))
            if not image_files:
                continue

            for img_file in image_files:
                try:
                    with st.spinner(f'Processing {os.path.basename(img_file)}...'):
                        image = Image.open(img_file)
                        ndvi, error = analyzer.process_ndvi_image(image)
                        if error:
                            continue

                        mean_ndvi, assessment, status = analyzer.analyze_ndvi(ndvi)

                        results.append({
                            "directory": dir_name,
                            "ndvi": mean_ndvi,
                            "assessment": assessment,
                            "status": status
                        })
                except Exception as e:
                    st.error(f"Error processing {img_file}: {str(e)}")
                    continue

    return results


def main():
    st.set_page_config(page_title="NDVI Analyzer", page_icon="?", layout="wide")

    st.title("? NDVI Vegetation Analyzer")

    # Sidebar
    st.sidebar.title("Analysis Options")
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode",
        ["Single Image", "Directory Analysis"]
    )

    # Initialize analyzer
    analyzer = NDVIAnalyzer()

    if analysis_mode == "Single Image":
        st.header("Single Image Analysis")

        # File uploader
        uploaded_file = st.file_uploader("Upload an image for analysis", type=['tif', 'png', 'jpg', 'jpeg'])

        # Visualization options
        viz_type = st.selectbox(
            "Select visualization type",
            ["standard", "heatmap"]
        )

        if uploaded_file is not None:
            try:
                # Process image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                with st.spinner('Processing image...'):
                    ndvi, error = analyzer.process_ndvi_image(image)

                    if error:
                        st.error(error)
                    else:
                        # Calculate metrics
                        mean_ndvi, assessment, status = analyzer.analyze_ndvi(ndvi)

                        # Display results
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Average NDVI", f"{mean_ndvi:.4f}")

                        with col2:
                            if status == "success":
                                st.success(assessment)
                            elif status == "warning":
                                st.warning(assessment)
                            else:
                                st.error(assessment)

                        # Show visualization
                        st.subheader("NDVI Distribution")
                        fig = analyzer.generate_visualization(ndvi, viz_type)
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:  # Directory Analysis
        st.header("Directory Analysis")

        directory = st.text_input("Enter directory path", "coldspringsfire")

        if st.button("Analyze Directory"):
            if os.path.exists(directory):
                results = process_directory(directory)

                if results:
                    st.subheader("Analysis Results")

                    # Create a DataFrame for better display
                    df = pd.DataFrame(results)

                    # Display summary statistics
                    st.write("Summary Statistics:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average NDVI", f"{df['ndvi'].mean():.4f}")
                    with col2:
                        st.metric("Number of Images", len(df))

                    # Display detailed results
                    st.write("Detailed Results:")
                    for result in results:
                        if result['status'] == 'success':
                            st.success(
                                f"Directory: {result['directory']}\n"
                                f"NDVI: {result['ndvi']:.4f}\n"
                                f"Assessment: {result['assessment']}"
                            )
                        elif result['status'] == 'warning':
                            st.warning(
                                f"Directory: {result['directory']}\n"
                                f"NDVI: {result['ndvi']:.4f}\n"
                                f"Assessment: {result['assessment']}"
                            )
                        else:
                            st.error(
                                f"Directory: {result['directory']}\n"
                                f"NDVI: {result['ndvi']:.4f}\n"
                                f"Assessment: {result['assessment']}"
                            )
                else:
                    st.warning("No results found in the specified directory")
            else:
                st.error("Directory not found")


if __name__ == "__main__":
    main()