# iSegMSI

iSegMSI provides a novel interactive segmentation strategy for MSI data. It could improve segmentation results by  subdividing or merging inappropriate regions into proper regions specified by scribbles. Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Overview of iSegMSI

<div align=center>
<img src="https://user-images.githubusercontent.com/70273368/172561018-b1ccd866-36b4-4890-9cb3-88e367191c19.png" width="600" height="500" /><br/>
</div>

__Schematic overflow of the iSegMSI model.__ (A) Architecture of the iSegMSI model. The method consists of two modules, including a dimensionality reduction (DR) module, and a feature clustering (FC) module that is consisted of a CNN block and an argmax classifier. (B) CNN block. Each of the first N-1 CNN components contain a 2D convolutional layer (p filters with 3×3 kernel size), a batch normalization, and a ReLU layer. The last component contains a 2D convolutional layer (q filters with 1×1 kernel size) and a batch normalization layer. 

# Requirement

    python == 3.5, 3.6 or 3.7
    
    pytorch == 1.8.2
    
    opencv == 4.5.3
    
    matplotlib == 2.2.2

    numpy >= 1.8.0
    
# Quickly start

## Preprocessing

Here, MSI data preprocessing including spectral alignment, peak detection, peak binning, peak filtering and peak pooling. Among them, spectral alignment, peak detection, peak binning are achieved using R package "MALDIquant", peak filtering and peak pooling are carried out by in-house Python scripts.

## Run iSegMSI model

cd to the iSegMSI fold

If you want to perfrom iSegMSI for unsupervised segmentation, taking fetus mouse data as an example, run:

    python run.py -input_file .../data/fetus_mouse.txt --input_shape 202 107 3 --use_scribble False --output_file output.txt

If you want to perfrom iSegMSI for interactive segmentation, taking fetus mouse data as an example, run:

    python run.py -input_file .../data/fetus_mouse.txt --input_shape 202,107,3 --use_scribble True -- input_scribble .../data/fetus_mouse_scribble.txt --output_file output.txt

    



