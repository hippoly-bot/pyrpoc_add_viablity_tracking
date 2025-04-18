# pyrpoc

Author: Ishaan Singh, Zhang Group (https://sites.google.com/view/zhangresearchgroup)
This software was written for the Zhang group's RPOC and SRS microscopy system, see https://www.nature.com/articles/s41467-022-32071-z. The purpose of it is to consolidate all the various LabVIEW scripts that are currently in use into a single all-in-one GUI with an intuitive interface. Functionalities include full flexibility of galvo mirror scanning, multi-channel imaging, multi-channel RPOC mask design and application, hyperspectral/z-stack imaging via a Zaber delay/Prior stage, and more. 
With any feedback or suggestions, please reach out to sing1125@purdue.edu.

## Basic Installation

The software is available as a package - make sure python 3.12 is in use, with a virtual environment if necessary. Once the virtual environment with python<=3.12 is created, run the following command. 

``` 
pip install pyrpoc
```

Then, run the command below to open the GUI. 

```
pyrpoc
```

If the command does not work, it is likely an issue with the PATH variable - make sure that Python scripts are available to PATH.
