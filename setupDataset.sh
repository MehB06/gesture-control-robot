#!/bin/bash

# This program downloads and setsup the ASL Alphabet Dataset from Kaggle

cd "$(pwd)"

# Permission to execute .sh file
echo "
This script will automatically:
  1). Download and extract the ASL Alphabeth Dataset (~1.1GB)
  2). Organise relevent folders into the correct directory.
  3). Delete extra redundent files.

Alternatively you can download it directly from 'https://www.kaggle.com/datasets/grassknoted/asl-alphabet'
 and organise the Class files into the 'data/' directory, as descibed in the README.md
"

read -p "Do you which to continue automatically? (y/n): " userInput

if [ "$userInput" != "y" ]; then
    echo "Aborting automatic install."
    exit 0
fi
echo ""

# Check if raw data already exists
if [ -d "data/raw" ]; then
    echo "Directory 'data/raw/' already exists.
    (If this is an error ensure that data/raw doesn't exist and try again.)
    Do you want to delete and reinstall or abort?"

    read -p "Do you want to reinstall the 'data/raw'? (y/n): " dirInput
    if [ "$dirInput" != "y" ]; 
        then
            echo "aborting Download"
            exit 0
        else
            rm -rf data/raw/*
    fi
fi
mkdir -p data/raw


# Download using cURL
echo "Downloading ASL Alphabet dataset..."
curl -L -o asl-alphabet.zip\
    https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet
echo "Dataset downloaded."
echo ""

# Extracting data and copying relevent data to the 'data/raw' directory
echo "Extracting Data..."
unzip -q asl-alphabet.zip -d temp/
echo "Dataset extracted."

if [ -d "temp/asl_alphabet_train/asl_alphabet_train" ]; 

    then
        echo "Moving data to correct directory..."
        mv temp/asl_alphabet_train/asl_alphabet_train/* data/raw/
        echo "Complete moving."

    else
        echo "Something when wrong... Aborting"
        rm -rf data/raw/*
        rm -rf temp/
        exit 0
fi
echo ""

echo "Cleaning up..."
rm -f asl-alphabet.zip
rm -rf temp/
echo "Setup Complete!"