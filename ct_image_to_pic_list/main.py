import numpy as np
from PIL import Image
from typing import Optional

def calculate_quartiles(arr):
    flattened = arr.flatten()
    q1 = np.percentile(flattened, 25, interpolation='linear')
    q3 = np.percentile(flattened, 75, interpolation='linear')
    return (q1, q3)

def convert_2darray_to_grey_image(arr: np.ndarray, min_val: Optional[float], max_val: Optional[float], lowerbound_show: Optional[float]) -> Image.Image:
    # Check if the input is a 2D array
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")
    
    Q1, Q3 = calculate_quartiles(arr) # Calculate the two quartiles

    # Normalize the array to the range 0-255 (handles input with arbitrary value ranges)
    # Outlier handling is required here
    if min_val is not None:
        arr_min = min_val
    else:
        arr_min = max(arr.min(), Q1 - 5.0 * (Q3 - Q1))
    
    if max_val is not None:
        arr_max = max_val
    else:
        arr_max = min(arr.max(), Q3 + 5.0 * (Q3 - Q1))
    
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
