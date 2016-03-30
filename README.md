# HPCDA
(High Performance Computing Data Analysis)

Overview
--------------
HPC Data Analysis is a project for understanding Supercomputers' memory access patterns and analyzing the data from running physical simulations on clusters. The project is based on python as well as its Scikit-learn, SciPy, MatPlotlib and SeaBorn libraries.

Directories
--------------
- **data**: The data folder contains all data csv files directly from Mitos or temporary data generated during the analysis process.
- **figs**: The figs folder contains all figures generated from the data analysis process.
- **src**: The src folder contains main.py as well as all other helper functions, models, and automations.
- **tsrc**: The tsrc folder contains some example usage of Scikit-learn machine-learning functions on the Titanic data set. (Unrelated to the project itself, but can be very useful to check some basic usage of Scikit-learn functions.)

Note
--------------
By running either the file **main.py** directly, or the file **ffsmetric.py**, some preprocessed data can be read in, and a plot called **kde_false_sharing.png** will be saved in the folder **figs**.

The regression model is stored in the file **regression.py**, and by running this file, we can check the factor of each metric that
contribute to the time latency for memory access (the higher the better a metric is). The use of metric is annotated in this file.

The *fuzzy-false-sharing-metric* based on the Kernel Density Estimation is stored in the file **ffsmetric.py**. It contains several slightly different variations of the metric ( first 10 version of them are deprecated ) which also differ in performance. (Currently, only the last version provides the proper evaluation and plot.)

There are also some test files ( with the format **test_i.py** ) in the **src** directory that are only for test and other trivaial purposes.

Plot (KDE False Sharing Metric)
--------------

![alt tag](https://raw.githubusercontent.com/xiawang/HPCDA/master/figs/figure_13_rt.png)