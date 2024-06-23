# Workforce_Attire_Project
`Authors: Kacper Grzymkowski, Adam Narożniak, Illia Tesliuk, Jakub Fołtyn`     
Project for detecting clothing items from photo.     
    
Main script to run the pipeline: ``run.py``. Example usage:   
`python run.py <folder with images> -o <output folder> -m <model name>`    
    
Arguments `<output folder>` and `<model name>` are optional. When output folder is not passed, the script will output JSON files to the console.   
   
Some example photos have been included in folder `\example_photos`

---------------------------------  
Software used for metric computations: [Object detection metrics](https://github.com/rafaelpadilla/review_object_detection_metrics)
