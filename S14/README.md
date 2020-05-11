**--------Back Ground Images-----------**


Total Number of Background Images -- 100. Extracted from google images using keywords, rooms, kitchen, amazon office and google office. 

**--------Foreground Images------------**


Total Number of Foreground Images -- 100. The Foreground images were also downloaded from google images but using keywords like man, woman, old couple, couple, and family
Steps used to create transparent foreground images --  
1. load data in GIMP 
2. desaturate the image 
3. Increase the color level to make sure people are darker than background.
4. Use brush to paint white over people and black over the back ground. 
5. then go to select all and cut from edit. 
6. Open the image again. Add the mask overlay. 
7. Go to Edit and paste the mask extracted in the step 5. 
8. Add fine touches by removing extra background left over 
9. Save Image as png file

**-------Background Foreground Images----**
Total 400,000 images. For every back ground image, all 100 foreground images were pasted on 20 random locations for 2 different orientations-- flipped horizontally and not flipped
Steps:
1. Open back ground image using PIL.Image.Open
2. resize the back ground image to 480x480. This was done to make sure we have squarish images and depth model outputs 240x240 depth map out.
If we had resized the back ground image to 240x240 instead, the depth model would have outputted 120x120. So, went ahead with 480x480 background image 
3. resize every foreground image to 240x240. This is to make sure that before pasting the foreground on background, we are making sure that the 
foreground isn't too big for the background image. 
4. Then for 20 different locations are randomly picked for where the foreground was pasted on top of back ground image using PIL.Image.paste method 
5. The step was performed for flipped orientation again.

**--------Masks Images---------**


Mask is available as 4th channel of png image. 
Total 400,000 images, same as background_foreground images
Along with all the steps mentioned above in foreground pasting on background, mask was extracted from 4th channel of foreground image.

**-------Depth Images ----------**


Depth Images were extracted from background foreground images created above using the depth wise model with nyu weights. 
The images were predicted in batches of 50 while creating background foreground images. 

File of code for dataset creation -- ![BackGround Foreground Images](https://github.com/rishubhkhurana/EVA/blob/master/S14/S14.ipynb)

The files are stored in 10 zip files within zip file created after processing 10 background images. 
Each zip file contains folders -- bg_images,bg_fg_images, bg_masks, bg_fg_depth_images

##############################################################################################

**Image statistics --**

File of code for statistics calculation -- ![BackGround Foreground Images](https://github.com/rishubhkhurana/EVA/blob/master/S14/S14_stats.ipynb)
Total Background Images: 100
Total Foreground Images: 100
Total Background Foreground Images: 400,000
Total Depth Images: 400,000
Total Masks Images: 400,000

Statistics--
Mean of Background Images: [148.93332309 136.86161233 125.61115295]
Std of Background Images: [64.58197291 64.13079164 68.30430912]

Mean of Background Foreground Images: [146.02142952 134.25154707 123.74065768]
Std of Background ForeGround Images: [66.4962467  65.56457192 69.03083189]

Mean of Masks Images: [10.38115305]
Mean of Masks Images: [49.05567378]

Mean of Depths Images: [34.15310015]
Std of Depths Images: [19.5175342]
#############################################################################################

**Examples**

Background Images

![BackGround Images](https://github.com/rishubhkhurana/EVA/blob/master/S14/background.PNG)

Background Foreground Images

![BackGround Foreground Images](https://github.com/rishubhkhurana/EVA/blob/master/S14/bg_fg.PNG)

Masks Images

![Masks Images](https://github.com/rishubhkhurana/EVA/blob/master/S14/bg_fg_masks.PNG)

Masks Images

![Depth Images](https://github.com/rishubhkhurana/EVA/blob/master/S14/depth.PNG)




