# Workforce_Attire_Project
`Authors: Kacper Grzymkowski, Adam Narożniak, Illia Tesliuk, Jakub Fołtyn`   

     
Project for detecting clothing items from photo. It utilizes a pipeline of first detecting persons from a photo and then detecting items of clothing. The pipeline is also capable of detecting a given item's color. 
# Files descriptions
This repository contains the following files:
- `Workforce_Attire_Project_final_report.pdf` -- file containing the final report for our project.
- `badge box detection.ipynb` -- notebook containing code for automatic detection of badges added by the [Ledits++ model](https://huggingface.co/spaces/editing-images/leditsplusplus).
- `fashionpedia_yolos_inference.ipynb` -- notebook containing examples of [YOLOS-small](https://huggingface.co/valentinafeve/yolos-fashionpedia) model inference.
- `get_clothing_color.ipynb` -- notebook containing code for finding the dominating clothing color.
- `human_clothes_detection.ipynb` -- notebook containing code for our pipeline (human and clothing detection).
- `human_clothes_detection_with_metric_computation.ipynb` -- notebook containing computed metrics for models with our pipeline.
- `metric_computation.ipynb` -- notebook containing computed metrics.
- `metric_computation_functions.py` -- file containing functions for computing metrics.
- `requirements.in` -- file containing only direct dependencies without set version numbers.
- `requirements.txt` -- file containing all dependencies, with version numbers.
- `run.py` -- **main script for running our clothing detection pipeline**.
- `spanish_dataset exploration.ipynb` -- notebook with preliminary EDA performed on the "Spanish" dataset.
- `yolo-fine-tune.ipynb` -- notebook containing code for YOLOS-tiny model fine-tuning.


Additionally, there are two folders:

- `\clothing_dataset` -- folder containing code for transforming the "Spanish" dataset into English (and bringing it into compliance with [Fashionpedia](https://huggingface.co/datasets/detection-datasets/fashionpedia)).
- `\example_photos` -- folder containing example photos for testing our algorithm.
# Example usage
Main script to run the pipeline: `run.py`. Example usage:   
`python run.py <folder with images> -o <output folder> -m <model name>`    
    
Arguments `<output folder>` and `<model name>` are optional. When output folder is not passed, the script will output JSON files to the console.   
   
Some example photos have been included in folder `\example_photos`

# Metric computation
We used averaged precision and recall for our models' evaluation. Moreover, for computing AP metric and precision-recall plots we used the following software:
[Object detection metrics](https://github.com/rafaelpadilla/review_object_detection_metrics)
