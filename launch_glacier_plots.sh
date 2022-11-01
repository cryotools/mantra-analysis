#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    echo
    echo "Usage: launch_glacier_plots.sh <TSL_input_file> <list_file> <step_size> <output_path>"  
    echo
elif [ ! -f $1 ]; then
    echo
    echo "Cannot open $1. Please provide a valid config file."
    echo
else

    # PATH DEFINITIONS
    # Modify according to your username and configuration
    WORK_PATH="<PATH_TO_STORE_STUFF>"
    OUTPUT_PATH="<PATH_TO_STORE_LOG_FILES>" 
    ERROR_PATH="<PATH_TO_STORE_ERROR_FILES>"      

    if [ ! -d "$WORK_PATH" ]; then mkdir -p $WORK_PATH; fi
    if [ ! -d "$OUTPUT_PATH" ]; then mkdir -p $OUTPUT_PATH; fi
    if [ ! -d "$ERROR_PATH" ]; then mkdir -p $ERROR_PATH; fi

    # PYTHON file
    # Path and name of the python file to be executed
    python_file="<REPO_PATH>/plot-glacier-timeseries.py"  

    ACCOUNT=morsanat

    input_file=$1

    list_file=$2
    # n_lines=6
    n_lines=$(wc -l < "$list_file")


    if [ -z ${3+x} ]; then 
        step=100
    else 
        step=$3
    fi
    
    output_path=$4

    i=0
    
    while [ $i -le $n_lines ]; do
	
        sbatch \
	    --ntasks=10 \
	    --output=$OUTPUT_PATH/%j.log \
	    --error=$ERROR_PATH/%j.log \
	    --workdir=$WORK_PATH \
	    --job-name="PPpy$i" \
	    --qos=short \
	    --exclusive \
	    --account=$ACCOUNT \
	    --partition=computehm \
	    ./PP_plt_glacier_TS.sh \
            $python_file \
            $input_file \
            $list_file \
            $i \
            $(( i + step )) \
            $output_path

        ((i+=step))
    done
    
fi











