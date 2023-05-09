# Source: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/

# Cmd to run: python CapstoneTrialTwo.py "C:\Users\budal\OneDrive\Documents\Capstone Research\Covid Scans\subject\98.12.2\*.dcm"

#Imports
import pydicom
import numpy as np
from stl import mesh
import sys
import glob
from skimage.measure import marching_cubes
import scipy.ndimage
import matplotlib.pyplot as plt

# Where data gets saved in between steps (saves processing time/effort)
output_path = working_path = "/Users/budal/OneDrive/Documents/Data/"

# This function is responsible for getting the .dcm files, counting them, storing them in an array 
# and determining data about them such as slice thickness

def load_dicom_image():
    print("Loading Dicom Image")
    files = []
    for filename in glob.glob(sys.argv[1], recursive=False):
        files.append(pydicom.dcmread(filename))

    print("file count: {}".format(len(files)))

    #Format slices
    slices = []
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
     
    #Slice thickness
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
    
# This function is responsible for gathering pixel data from the .dcm files, and converting 
# the pixel data into HU
def get_pixel_data(slice_set):
     print("Getting Pixel Data")
     image = np.stack([s.pixel_array for s in slice_set])
     image = image.astype(np.int16)
     
     # All pixels @2000 value == 0
     image[image == -2000] = 0
     
     #Calculation for pixel -> HU
     intercept = slice_set[0].RescaleIntercept
     slope = slice_set[0].RescaleSlope
     if slope != 1:
         image = slope * image.astype(np.float64)
         image = image.astype(np.int16)
     image += np.int16(intercept)
     patient_HU = np.array(image, dtype=np.int16)
     
     return patient_HU
  
# This function is responible for setting the spacing between slices
def stack_create(image, scan, new_spacing = [1,1,1]):
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
  
    return image, new_spacing

# This function is responsible for setting the threshold value to show only bone, and 
# transposing the image matrix
def make_mesh(image):
    threshold=300
    step_size=1
    transposed = image.transpose(0,2,1)
    transpose = transposed.transpose(2,1,0)
    verts, faces, norm, val = marching_cubes(transpose, threshold, step_size=step_size, allow_degenerate=True)
    
    return verts, faces

# Main Section resposible for running the program
#%%

# Convert to .stl and export

# Convert to .stl format
id = 0
dicom_slices = load_dicom_image()
patient_HU = get_pixel_data(dicom_slices)
np.save(output_path + "fullimages_%d.npy" % (id), patient_HU)

# Show the graph of the housefield units (HU)
plt.hist(patient_HU.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

id = 0
patient_HU = np.load(output_path+'fullimages_{}.npy'.format(id))
resampled_patient, new_spacing = stack_create(patient_HU, dicom_slices)
v, f = make_mesh(resampled_patient)
body = mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype))

#%%
for i, fc in enumerate(f):
    for j in range(3):
        body.vectors[i][j] = v[fc[j],:]
body.save('spinemodel.stl')
