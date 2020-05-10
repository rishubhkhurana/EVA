--------Back Ground Images-----------
Total Number of Background Images -- 100. Extracted from google images using keywords, rooms, kitchen, amazon office and google office. 

--------Foreground Images------------
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
-------Background Foreground Images----
Total 400,000 images. For every back ground image, all 100 foreground images were pasted on 20 random locations for 2 different orientations-- flipped horizontally and not flipped
Steps:
1. Open back ground image using PIL.Image.Open
2. resize the back ground image to 480x480. This was done to make sure we have squarish images and depth model outputs 240x240 depth map out.
If we had resized the back ground image to 240x240 instead, the depth model would have outputted 120x120. So, went ahead with 480x480 background image 
3. resize every foreground image to 240x240. This is to make sure that before pasting the foreground on background, we are making sure that the 
foreground isn't too big for the background image. 
4. Then for 20 different locations are randomly picked for where the foreground was pasted on top of back ground image using PIL.Image.paste method 
5. The step was performed for flipped orientation again.

--------Masks Images---------
Mask is available as 4th channel of png image. 
Total 400,000 images, same as background_foreground images
Along with all the steps mentioned above in foreground pasting on background, mask was extracted from 4th channel of foreground image.

-------Depth Images ----------
Depth Images were extracted from background foreground images created above using the depth wise model with nyu weights. 
The images were predicted in batches of 50 while creating background foreground images. 

File of code -- S14.ipynb


Image statistics --
File of code -- S14_stats.ipynb




Examples

Background Images

![BackGround Images](https://github.com/rishubhkhurana/EVA/blob/master/S14/background.PNG)

Background Foreground Images

![BackGround Foreground Images](https://github.com/rishubhkhurana/EVA/blob/master/S14/bg_fg.PNG)








