NOTE: This is a copy of https://github.com/vacancy/PyPatchMatch, with some code related to travis removed

PatchMatch based Inpainting
=====================================
This library implements the PatchMatch based inpainting algorithm. It provides both C++ and Python interfaces.
This implementation is heavily based on the implementation by Younesse ANDAM:
(younesse-cv/PatchMatch)[https://github.com/younesse-cv/PatchMatch], with some bugs fix.

Usage
-------------------------------------

You need to first install OpenCV to compile the C++ libraries. Then, run `make` to compile the
shared library `libpatchmatch.so`.

For Python users (example available at `examples/py_example.py`)

```python
import patch_match

image = ...  # either a numpy ndarray or a PIL Image object.
mask = ...   # either a numpy ndarray or a PIL Image object.
result = patch_match.inpaint(image, mask, patch_size=5)
```

For C++ users (examples available at `examples/cpp_example.cpp`)

```cpp
#include "inpaint.h"

int main() {
    cv::Mat image = ...
    cv::Mat mask = ...

    cv::Mat result = Inpainting(image, mask, 5).run();

    return 0;
}
```


README and COPYRIGHT by Younesse ANDAM
-------------------------------------
@Author: Younesse ANDAM

@Contact: younesse.andam@gmail.com

Description: This project is a personal implementation of an algorithm called PATCHMATCH that restores missing areas in an image.
The algorithm is presented in the following paper
 PatchMatch  A Randomized Correspondence Algorithm
               for Structural Image Editing
   by C.Barnes,E.Shechtman,A.Finkelstein and Dan B.Goldman
   ACM Transactions on Graphics (Proc. SIGGRAPH), vol.28, aug-2009

 For more information please refer to
 http://www.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php

Copyright (c) 2010-2011


Requirements
-------------------------------------

To run the project you need to install Opencv library and link it to your project.
Opencv can be download it here
http://opencv.org/downloads.html

