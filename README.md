# iSegMSI

iSegMSI provides a novel interactive segmentation strategy for MSI data. It could improve segmentation results by  subdividing or merging inappropriate regions into proper regions specified by scribbles. Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Overview of dc-DeepMSI

<div align=center>
<img src="https://user-images.githubusercontent.com/70273368/172561018-b1ccd866-36b4-4890-9cb3-88e367191c19.png" width="600" height="500" /><br/>
</div>

__Schematic overflow of the iSegMSI model.__ (A) Architecture of the iSegMSI model. The method consists of two modules, including a dimensionality reduction (DR) module, and a feature clustering (FC) module that is consisted of a CNN block and an argmax classifier. (B) CNN block. Each of the first N-1 CNN components contain a 2D convolutional layer (p filters with 3×3 kernel size), a batch normalization, and a ReLU layer. The last component contains a 2D convolutional layer (q filters with 1×1 kernel size) and a batch normalization layer. 

# Requirement

    python == 3.5, 3.6 or 3.7
    
    pytorch == 

    numpy >= 1.8.0
    

# Run iSegMSI model



