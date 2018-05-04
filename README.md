# EC504 Project

This is the repository for Boston University EC504 project _Image Segmentation via Network Flow_.

## File Sescription

| Filename | Description | 
| --- | --- |
| `bayes.py` | Naive Bayes classifier class |
| `ek.py` | Edmonds-Karp algorithm class |
| `main_gmm.py` | Main program in GMM mode |
| `main_kmean.py` | Main program in k-mean mode |
| `main_ui.py` | Main program used for GUI extension |
| `ui.py` | GUI program |
| `requirement.txt` | Requirement packages list for `pip` |

| Foldername | Description | 
| --- | --- |
| `cython` | Folder contains Cython files |
| `images` | Folder contains test images |
| `reports` | Folder contains reports |

## Before Using

Don't forget install the required libraries by command:

```
$ sudo pip install -r requirement.txt
```

The required Python libraries are:

| Name | Version | Description |
| --- | --- | --- |
| OpenCV | 3.4.0.12 | Computer vision library |
| Matplotlib | 2.1.2 | 2D plotting library |
| NumPy | 1.12.0 | Scientific computing library |
| SciPy | 1.0.0 | Scientific computing library |
| PyQt5 | 5.10.1 | User interface library |
| Cython | 0.28.2 | Cython library |

## Usage

### main_gmm.py

```
$ python main_gmm.py [image_name] [scale_factor]
```

* ```image_name```: The name for the image to be segmented.

* ```scale_factor```: The factor that you want the image to be shrinked, typically larger factor will cost more time to run, even have memory errors. For test images we provide, best scale is 0.2 - 0.25.

### main_kmean.py

```
$ python main_kmean.py [image_name] [scale_factor]
```

* ```image_name```: The name for the image to be segmented.

* ```scale_factor```: The factor that you want the image to be shrinked, typically larger factor will cost more time to run, even have memory errors. For test images we provide, best scale is 0.2 - 0.25.

### ui.py

```
$ python ui.py
```

If not works, try:

```
$ pythonw ui.py
```

### Cython

```
$ sh ./start.sh
```
