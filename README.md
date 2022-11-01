# MANTRA Analysis
Scripts to analyze data generated using the ["MountAiN glacier Transient snowline Retrieval Algorithm"](https://github.com/cryotools/mantra). The data must have been postprocessed using the [MANTRA Postprocessing scripts](https://github.com/cryotools/mantra-postprocessing) prior to analysis.

In case you have a Slurm-based HPC cluster at hand, you may use the Bash shell files to run scripts in parallel.

## Requirements

### Data

First, you will need to process some [MANTRA TSLA data](https://github.com/cryotools/mantra). An small example file is `./data/MANTRA/`.

Many of the scripts require a CSV file of Randolph Glacier Inventory (RGI) data to work. An example for High Mountain Asia is provided in `./data/RGI`. For other regions, you may [download the according shapefile](https://nsidc.org/data/nsidc-0770/versions/6), open it in your preferred GIS environment, and export the attribute table to CSV.

### Software

- Python 3 (a current version of [Anaconda](https://www.anaconda.com/) is recommended).
- Pandas, Numpy, MatPlotLib and quite a lot of other Python libs (that will come with Anaconda).
- Some scripts use [Scientific Color Maps v6](https://zenodo.org/record/4153113).

# Scripts

### annual_maxima.py
Create a stacked bar plot displaying the number of glaciers that have their maxima in each year.

### EOF-analysis.py
Empirical Orthognal Function analysis of spatio-temporal patterns.

### tsl-full-timeseries-plots.py
Plot full TSLA dataset into one time series figure.

### plot-glacier-timeseries.py
Plot TSL timeseries for an individual glacier.

```console
python plot-glacier-timeseries.py <tsl_file> <glacier_list_file> <lower_limit> <upper_limit> <output_dir>\n\n")    
```
With `tsl_file` being a MANTRA TSL result file in HDF format, `glacier_list_file` an ASCII text file containing the RGI_IDs, one ID per row, `lower_limit` and `upper_limit` the first and last dataset to process, referring to number of IDs in glacier_list_file, and `output_dir` a valid directory to which the output will be written.



# Citing MANTRA
If you publish work based on MANTRA, please cite as:

David Loibl (2022): MountAiN glacier Transient snowline Retrieval Algorithm (MANTRA) v0.8.2, doi: [10.5281/zenodo.7133644](https://doi.org/10.5281/zenodo.7133644)

# Acknoledgements
MANTRA was developed within the research project "TopoClimatic Forcing and non-linear dynamics in the climate change adaption of glaciers in High Asia" (TopoClif). TopoCliF and David Loibl's work within the project were funded by [DFG](https://gepris.dfg.de/gepris/projekt/356944332) under the ID LO 2285/1-1.
