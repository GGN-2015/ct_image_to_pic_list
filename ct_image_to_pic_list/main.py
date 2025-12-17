import numpy as np
from PIL import Image
from typing import Optional

def calculate_percentile(arr):
    flattened = arr.flatten()
    p01 = np.percentile(flattened, 1, method='linear')
    p99 = np.percentile(flattened, 99, method='linear')
    return (p01, p99)

def convert_2darray_to_grey_image(arr: np.ndarray, min_val: Optional[float], max_val: Optional[float], lowerbound_show: Optional[float]) -> Image.Image:
    # Check if the input is a 2D array
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")
    
    P01, P99 = calculate_percentile(arr) # Calculate the two quartiles

    # Normalize the array to the range 0-255 (handles input with arbitrary value ranges)
    # Outlier handling is required here
    if min_val is not None:
        arr_min = min_val
    else:
        arr_min = P01
    
    if max_val is not None:
        arr_max = max_val
    else:
        arr_max = P99
    
    # Unify the value range
    arr = arr.copy()
    arr[arr > arr_max] = arr_max
    arr[arr < arr_min] = arr_min

    if lowerbound_show is not None: # Set the focus lower bound
        arr[arr < lowerbound_show] = min_val

    if arr_max == arr_min: # Avoid division by zero; set all values to 0 (black) if they are identical
        normalized = np.zeros_like(arr, dtype=np.uint8)
    else:
        normalized = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    
    # Create a grayscale image (Mode 'L' represents 8-bit grayscale)
    img = Image.fromarray(normalized, mode='L')
    return img

def convert_3darray_to_grey_image_list(arr3d: np.ndarray, min_val: Optional[float], max_val: Optional[float], lowerbound_show: Optional[float]) -> list[Image.Image]:
    # Check if the input is a 3D array
    if arr3d.ndim != 3:
        raise ValueError("Input must be a 3D numpy array")
    
    ans = []

    # Slice along the last dimension
    # Equivalent to obtaining cross-sections
    for slice_id in range(arr3d.shape[2]):
        img = convert_2darray_to_grey_image(arr3d[:,:,slice_id], min_val, max_val, lowerbound_show)
        ans.append(img)

    return ans

def convert_ct_3darray_to_grey_image_list(arr3d: np.ndarray, min_val: Optional[float]=-200, max_val: Optional[float]=300, lowerbound_show: Optional[float]=None) -> list[Image.Image]:
    return convert_3darray_to_grey_image_list(arr3d, min_val, max_val, lowerbound_show)

def convert_mri_3darray_to_grey_image_list(arr3d: np.ndarray, min_val: Optional[float]=30, max_val: Optional[float]=None, lowerbound_show: Optional[float]=None) -> list[Image.Image]:
    return convert_3darray_to_grey_image_list(arr3d, min_val, max_val, lowerbound_show)
