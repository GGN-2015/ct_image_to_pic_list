# ct_image_to_pic_list
convert a 3d CT image (numpy) into pillow image list.

## Installation
```bash
pip install ct_image_to_pic_list
```

## Usage
```python
import numpy as np
from ct_image_to_pic_list import convert_3darray_to_grey_image_list

# construct you 3d CT image (HU value)
arr3d = np.array([ ... ])

# get a list of PIL.Image.Image
img_list = convert_3darray_to_grey_image_list(arr3d)

# save it if you want
for index, img in enumerate(img_list):
    img.save(f"{index:05d}.png")
```
