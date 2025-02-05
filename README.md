# pysrs
A python package for coordinating DAQ for stimulated Raman spectroscopy. Project start date: 1/15/24

Author: Ishaan Singh, Zhang Group (https://sites.google.com/view/zhangresearchgroup)

## Basic Installation

Navigate to the directory that contains ```setup.py```. Run 

``` 
pip install -e .
```

## Current Features
- All-in-one GUI for performing Raman imaging with the instruments in-lab, modularized for flexibility when instruments change or for other labs to adopt
- Coordinated instruments: analog outputs/inputs (galvo mirrors and lock-in amplifier respectively), Zaber movable delay stage, Prior ProScan3 movable stage
- Support for the real-time precision opto-control technique previously developed by the Zhang group

Please reach out to sing1125@purdue.edu with any suggestions or feedback on the GUI. 
