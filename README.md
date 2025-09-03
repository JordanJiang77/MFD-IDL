# MFD-IDL
This repository contains the official code and datasets for **A novel hybrid macroscopic fundamental diagram-informed deep learning method for lane-level traffic prediction** 


**Link** https://doi.org/10.1016/j.inffus.2025.103655

**Authors: ** Ruiyuan Jiang, Shangbo Wang,  Bingyi Liu, Yuli Zhang, Pengfei Fan, Dongyao Jia 
 

We would appreciate a citation of our work: Ruiyuan Jiang, Shangbo Wang, Bingyi Liu, Yuli Zhang, Pengfei Fan, Dongyao Jia, A novel hybrid macroscopic fundamental diagram-informed deep learning method for lane-level traffic prediction, Information Fusion, Volume 126, Part B,
2026, 103655, ISSN 1566-2535, https://doi.org/10.1016/j.inffus.2025.103655.



# How to run
**1. Download the dataset**
The dataset of I-24 Motion is available at: https://i24motion.org/data
The trajectory data can be aggregated to any fine-grained traffic data in any time and space through macro.py.  
**2. Process and Run**
1. MFD-IDL.py is the main python code for lane-level traffic flow prediction.
2. Any fine-grained traffic state can be aggregated from macro.py.
3. Parameters of MFD can be calibrated by mfd_calibration.py.
