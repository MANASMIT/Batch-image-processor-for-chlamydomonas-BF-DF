import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import color, filters, measure
from skimage.measure._regionprops import RegionProperties
import os
import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image
from openpyxl import Workbook
import base64
plt.rcParams['figure.max_open_warning'] = 50
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

# Define sphericity and size cutoffs (you can adjust these)
min_sphericity = 0.08
min_size = 500
average_cell_size = 7000
max_size = 25000
min_intensity = 0.40



def process_image(img_path):

    min_sphericity = 0.08
    min_size = 500
    average_cell_size = 7000
    max_size = 25000
    min_intensity = 0.40

    # Load the image (replace with your image path)
    im = io.imread(img_path, as_gray=True)
    im = im[10:-10, 10:-10]

    im = filters.gaussian(im, sigma=2)

    # Invert the image so that cells are white and background is black
    #im = np.invert(im)

    # Threshold the image to get a binary image with cells as 1s and background as 0s
    thresh = filters.threshold_otsu(im)
    binary = im < thresh

    # Label the binary image to identify connected components (i.e., cells)
    label_image = measure.label(binary, connectivity=1)

    #Keep count of the number of cells that meet your criteria
    cell_count = 0
    cells_over_avg_size =0
    total_area = 0

    # Filter regions based on size and sphericity
    cells = []
    new_label = 1
    for region in measure.regionprops(label_image, intensity_image=im):
        if region.area > min_size and region.area < max_size:
            equivalent_diameter = region.equivalent_diameter
            perimeter = region.perimeter
            sphericity = 4 * np.pi * region.area / (perimeter ** 2)
            mean_intensity = region.mean_intensity
            if sphericity > min_sphericity and region.coords[:, 0].min() > 0 and region.coords[:, 0].max() < im.shape[0] - 1 and region.coords[:, 1].min() > 0 and region.coords[:, 1].max() < im.shape[1] - 1 and mean_intensity > min_intensity:
                if region.area > 14000:
                    #print (f'{style.RED}{region.area}{style.RESET}')
                    n_cells = round(region.area / average_cell_size)
                    cells+=[region]*n_cells # add the original region n_cells times
                    cell_count += n_cells
                else:
                    #print(region.area)
                    cells.append(region)
                    cell_count += 1
                if region.area > average_cell_size:
                    cells_over_avg_size += 1

           #for cell in cells:
                #total_area += cell.area
            #average_cell_size = total_area / cell_count

    # Create a plot of the image with cells overlaid

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.imshow(binary, cmap='gray')
    ax1.set_title('Thresholded image')
    ax2.imshow(im)
    ax2.set_title(f'Number of cells: {len(cells)},Cells over avg size: {cells_over_avg_size} ')
    for cell in cells:
        ax2.annotate('', xy=(cell.centroid[1], cell.centroid[0]), xytext=(cell.centroid[1] + 10, cell.centroid[0] + 10),
                     color='red', arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->"))

    # Convert the plot to a base64-encoded string

    #print(f'{style.RED}Average cell size:, {style.RED}{average_cell_size}{style.RESET}')
    #Create a dataframe with the cell properties
    data = {'Image': [os.path.basename(img_path)]*len(cells),
            'Cell_Number': [i + 1 for i in range(cell_count)],
            'Cell_Area': [cell.area for cell in cells],
            'Equivalent_Diameter': [cell.equivalent_diameter for cell in cells],
            'Perimeter': [cell.perimeter for cell in cells],
            'Sphericity': [4*np.pi * cell.area / (cell.perimeter ** 2) for cell in cells],
            'Mean_Intensity': [cell.mean_intensity for cell in cells],
            'Cells_over_Avg_Size':[ cells_over_avg_size for cell in cells]}

    # Save the plot as a PNG file
    plot_file = os.path.splitext(os.path.basename(img_path))[0] + '.png'
    new_path = "C:/Users/m_jen/Downloads/Sem 8/Sen lab project/MJ/Images/Cr dcfda CLONE 3 AND 189/Pre Cu Stress DCFDA/BF/check" + "/" + plot_file
    plt.savefig(new_path, dpi=300, bbox_inches='tight')

    with open(new_path, 'rb') as f:
        img_data = f.read()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    dfk = pd.DataFrame(data)
    return dfk, img_base64

def process_directory(dir_path, output_file):
    dfs = []
    img_base64s = []

    # Loop through each file in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file is an image
        if file_name.endswith('.tif') or file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Get the full path to the image file
            img_path = os.path.join(directory_path, file_name)
            # Process the image and get the dataframe
            df,img_base64 = process_image(img_path)
            # Append the dataframe to the list
            dfs.append(df)
            img_base64s.append(img_base64)

    # Concatenate all the dataframes into a single dataframe
    result_df = pd.concat(dfs,axis=0, ignore_index=True)

    # Concatenate all the image strings into a single string
    result_img_base64 = ''.join(img_base64s)

    # Save the result dataframe and image string to the output file
    with open(output_file, 'w') as f:
        # Write the dataframe to the file as CSV
        result_df.to_csv(f, index=False)
        # Write the image string to the file
        f.write(result_img_base64)

    # Save the dataframe to a file if an output file is specified
    if output_file:
        result_df.to_excel(output_file, index=False)
    return result_df

output_file = 'C:/Users/m_jen/Downloads/Sem 8/Sen lab project/MJ/output2.xlsx'
directory_path = 'C:/Users/m_jen/Downloads/Sem 8/Sen lab project/MJ/Images/Cr dcfda CLONE 3 AND 189/Pre Cu Stress DCFDA/BF'
result_df = process_directory(directory_path, output_file)


    # Display the thresholded image and final image with counted cells
    #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    #ax1.imshow(binary, cmap='gray')
    #ax1.set_title('Thresholded image')
    #ax2.imshow(im)
    #ax2.set_title(f'Number of cells: {len(cells)},Cells over avg size: {cells_over_avg_size} ')
    #for cell in cells:
        #ax2.annotate('', xy=(cell.centroid[1], cell.centroid[0]), xytext=(cell.centroid[1]+10, cell.centroid[0]+10), color='red', arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->"))



    # Close plot
   # plt.close()

print('Number of chlamydomonas cells: ', cell_count, average_cell_size )
