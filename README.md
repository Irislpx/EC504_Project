# EC504_Project

This is the repository for Boston University EC504 project _Image Segmentation via Network Flow_.

## File Sescription

| Filename | Description | 
| --- | --- |
| `bayes.py` | Naive Bayes classifier class |
| `ek.py` | Edmonds-Karp algorithm class |
| `main.py` | Main program in annotation mode |
| `main_kmean.py` | Main program in k-mean mode |
| `requirement.txt` | Requirement packages list for `pip` |
| `teagle.jpg` | Test image |
| `teagle_anno.jpg` | Test image annotation |

## Before Using

Don't forget install the required libraries by command:

```
pip install -r requirement.txt
```

The required Python libraries are:

| Name | Version | Description |
| --- | --- | --- |
| OpenCV | 3.4.0.12 | Computer vision library |
| Matplotlib | 2.1.2 | 2D plotting library |
| NumPy | 1.12.0 | Scientific computing library |
| SciPy | 1.0.0 | Scientific computing library |

## Usage

### main.py

```
python main.py [image_name] [annotation_name]
```

* ```image_name```: The name for the image to be segmented.

* ```annotation_name```: The name for the annotation image with blue lines indicate background and red lines indicate foreground, the size should be the same as the image to be segmented.

### main_kmean.py

```
python main_kmean.py [image_name]
```

* ```image_name```: The name for the image to be segmented.
