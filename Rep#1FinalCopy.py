import os
os.environ['MPLBACKEND'] = 'Agg'  # Set the matplotlib backend before importing pyplot
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image
from pathlib import Path
import io

st.title("Synthetic NDVI Converter for Drought Estimation")
st.write("""
This app converts RGB satellite images to synthetic NDVI images.
You can either upload a single image or process an entire directory of images. Think of google maps images, but now you can turn them into synthetic NDVI images for drought estimation
""")

def calculate_synthetic_ndvi(image_array):
    """Calculate synthetic NDVI from an RGB image array."""
    image_float = image_array.astype(float)
    r = image_float[:, :, 0]
    g = image_float[:, :, 1]
    b = image_float[:, :, 2]
    
    # Approximate NIR using average of RGB channels
    nir_approx = (r + g + b) / 3
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Calculate synthetic NDVI
    ndvi = (nir_approx - r) / (nir_approx + r + epsilon)
    
    # Normalize to 0-1 range
    ndvi = (ndvi + 1) / 2
    
    return ndvi

def process_single_image(image):
    """Process a single image to generate synthetic NDVI."""
    image_array = np.array(image)
    ndvi = calculate_synthetic_ndvi(image_array)
    return ndvi

def visualize_ndvi_distribution(ndvi, filename=None):
    """
    Create a Seaborn visualization of NDVI value distribution.
    
    Parameters:
    ndvi (numpy.ndarray): Synthetic NDVI image array
    filename (str, optional): Name of the image file
    
    Returns:
    tuple: (matplotlib.figure.Figure, pandas.DataFrame)
    """
    # Flatten the NDVI array for distribution plotting
    ndvi_flat = ndvi.flatten()
    
    # Create a DataFrame for more detailed analysis
    ndvi_df = pd.DataFrame({
        'NDVI_Values': ndvi_flat
    })
    
    # Compute detailed statistics
    ndvi_stats = ndvi_df['NDVI_Values'].describe()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram with Seaborn
    sns.histplot(ndvi_df['NDVI_Values'], kde=True, color='skyblue', ax=ax1)
    ax1.set_title(f'NDVI Value Distribution\n{filename or ""}')
    ax1.set_xlabel('NDVI Value')
    ax1.set_ylabel('Frequency')
    
    # Box plot with Seaborn
    sns.boxplot(x=ndvi_df['NDVI_Values'], color='lightgreen', ax=ax2)
    ax2.set_title('NDVI Value Box Plot')
    ax2.set_xlabel('NDVI Value')
    
    # Add statistical annotations
    stats_text = (
        f"Mean NDVI: {ndvi_stats['mean']:.4f}\n"
        f"Median NDVI: {ndvi_stats['50%']:.4f}\n"
        f"Min NDVI: {ndvi_stats['min']:.4f}\n"
        f"Max NDVI: {ndvi_stats['max']:.4f}\n"
        f"Standard Deviation: {ndvi_stats['std']:.4f}"
    )
    plt.figtext(0.5, -0.05, stats_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    return fig, ndvi_stats

def main():
    # File uploader for single image
    uploaded_file = st.file_uploader(
        "Select an RGB satellite image of a landscape of your choosing",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff']
    )
    
    if uploaded_file:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process and display NDVI
        ndvi = process_single_image(image)
        
        # Display NDVI image using a colormap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title("Synthetic NDVI Image")
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        plt.close(fig)  # Close the figure to free up memory
        
        # Visualize NDVI Distribution
        distribution_fig, ndvi_stats = visualize_ndvi_distribution(ndvi, uploaded_file.name)
        st.pyplot(distribution_fig)
        plt.close(distribution_fig)
        
        # Display NDVI statistics as a DataFrame
        st.write("NDVI Statistics:")
        st.dataframe(ndvi_stats)
        
        # Option to download NDVI statistics
        ndvi_stats_df = pd.DataFrame([ndvi_stats])
        csv = ndvi_stats_df.to_csv(index=False)
        st.download_button(
            label="Download NDVI Statistics",
            data=csv,
            file_name=f"ndvi_stats_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )
        
        # Download button for the NDVI image
        buf = io.BytesIO()
        plt.figure(figsize=(8, 6))
        plt.imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1)
        plt.title("Synthetic NDVI Image")
        plt.axis('off')
        plt.colorbar(shrink=0.8)
        plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.1)
        plt.close()  # Close this figure as well
        
        st.download_button(
            label="Download NDVI Image",
            data=buf.getvalue(),
            file_name="ndvi_result.png",
            mime="image/png"
        )
    
    # Directory processing section
    directory = st.text_input("Enter directory path for batch processing (.jpg, .jpeg, .png, .tiff, .tif)")
    
    def process_directory(directory_path):
        """Process all images in a given directory."""
        processed_images = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        
        for file_path in Path(directory_path).glob('*'):
            if file_path.suffix.lower() in valid_extensions:
                try:
                    image = Image.open(file_path)
                    ndvi = process_single_image(image)
                    processed_images.append((file_path.name, ndvi))
                except Exception as e:
                    st.error(f"Error processing {file_path.name}: {str(e)}")
        
        return processed_images
    
    if directory and st.button("Process Directory"):
        if os.path.isdir(directory):
            processed_images = process_directory(directory)
            
            if processed_images:
                st.success(f"Processed {len(processed_images)} images")
                
                # Create output directory
                output_dir = os.path.join(directory, "ndvi_results")
                os.makedirs(output_dir, exist_ok=True)
                
                # Prepare to collect NDVI statistics across all images
                all_ndvi_values = []
                ndvi_stats_list = []
                
                # Save processed images
                for filename, ndvi in processed_images:
                    # Flatten NDVI values for overall statistics
                    ndvi_flat = ndvi.flatten()
                    all_ndvi_values.extend(ndvi_flat)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1)
                    ax.set_title(f"NDVI for {filename}")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    
                    # Save the figure
                    output_path = os.path.join(output_dir, f"ndvi_{filename.split('.')[0]}.png")
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)  # Close the figure to free up memory
                    
                    # Compute and store statistics for each image
                    _, image_stats = visualize_ndvi_distribution(ndvi, filename)
                    ndvi_stats_list.append({
                        'Filename': filename,
                        **image_stats
                    })
                
                # Create directory-wide distribution plot
                if all_ndvi_values:
                    directory_ndvi = np.array(all_ndvi_values)
                    dir_distribution_fig, _ = visualize_ndvi_distribution(directory_ndvi, "All Images")
                    st.pyplot(dir_distribution_fig)
                    plt.close(dir_distribution_fig)
                
                # Create a DataFrame of NDVI statistics and display
                ndvi_stats_dataframe = pd.DataFrame(ndvi_stats_list)
                st.write("NDVI Statistics for All Images:")
                st.dataframe(ndvi_stats_dataframe)
                
                # Option to download full statistics
                csv = ndvi_stats_dataframe.to_csv(index=False)
                st.download_button(
                    label="Download Full NDVI Statistics",
                    data=csv,
                    file_name="ndvi_directory_stats.csv",
                    mime="text/csv"
                )
                
                st.success(f"Results saved to {output_dir}")
            else:
                st.warning("No valid images found")
        else:
            st.error("Invalid directory path")

# Run the main function
if __name__ == "__main__":
    main()
