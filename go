#!/bin/bash  

export MALLOC_TRIM_THRESHOLD_=65536

# Define the directory name
DIR="simulation"

# Check if the directory exists
if [ -d "$DIR" ]; then
    # Directory exists, so remove all files inside it
    rm -rf "$DIR"/*
else
    # Directory doesn't exist, so create it
    mkdir "$DIR"
fi

rm cropped/*
date

./electromag.py


