# Tracking Hand  
This is a project about tracking left hand and right hand for sign language estimation: <https://github.com/PohanYang/Sign-Language-Estimation-by-Tensorflow>  
  
  
Demo Video: <https://youtu.be/enSGG8Od8uM>  
Use Language: Python 2.7  
Use Tool: OpenCV  
  
HSV is the most common cylindrical-coordinate representations of points in an RGB color model, stands for H(hue), S(saturation), and V(value). The Results for the HSV skin detection color based algorithm pf [5] (where parameter values are [thRegion, thNeighbour, thRGB] = [0.4, 0.1, 0.5]), I fix parameter values to 0 < H < 50, and 0.23 < S < 0.68 is more popular parameter in skin detection now.  
It is tracking right hand and left hand, Marked right hand on RED and left hand on BLUE, use skin detection algorithm [2] from webcam to segement skin. Figure 3 show the same image of figure 2 convert from RGB to HSV.  
![fig1](https://github.com/PohanYang/Tracking-Hand/blob/master/img/wikihsv.png)  
Figure 1. HSV(picture from wiki)  
![fig2](https://github.com/PohanYang/Tracking-Hand/blob/master/img/orgpic.PNG)  
Figure 2. Original Picture.  
![fig3](https://github.com/PohanYang/Tracking-Hand/blob/master/img/hsv.PNG)  
Figure 3. Convert figure 2 fram RGB to HSV.  
Figure 4 illustrates an final image, where all pixels classified as skin (using the range in channel H already established) were set to value 255, and non-skin pixels was fixed to 0.  
Figure 4. classified skin set to value 255.  
![fig4](https://github.com/PohanYang/Tracking-Hand/blob/master/img/bw.PNG)  
Now we have pixels value 0 in figure 4, that skin in this picture, next step I want to remove face. I use [6] Haar feature-based cascade classifiers to find where face is.  
In [7] we will see the basics of face detection using Haar Feature-based Cascade Classifiers, the word "Cascade" is means that the resultant classifier consists of several simpler classifiers that are applied subsequently to a region of interest until at some stage the candidate is rejected or all the stages are passed.  
The feature used in a particular classifier is specified by its shape , position within the region of interest and the scale. After we use haar feature-based cascade classifier we can get hand shape in figure 6.  
![fig5](https://github.com/PohanYang/Tracking-Hand/blob/master/img/haarfeatures.png)  
Figure 5. Features in Haar Cascade Classifier, in 1 is edge features, 2 is line features, and 3 is center surround features.  
![fig6](https://github.com/PohanYang/Tracking-Hand/blob/master/img/noface.png)  
Use Haar Feature-based Cascade Classifier remove face from figure 4.  
  
Reference:  
[1] Hand tracking and gesture recognition system for human-computer interaction using low-cost hardware <https://link.springer.com/article/10.1007/s11042-013-1501-1>  
[2] Skin Detection using HSV color space - V. A. Oliveira, A. Conci  <http://webcache.googleusercontent.com/search?q=cache:http://www.matmidia.mat.puc-rio.br/sibgrapi/media/posters/59928.pdf&gws_rd=cr&ei=i87IWPa3LMuF8wWI05KgAw>  
[3] OpenCv Documentation  
[4] RGB-H-CbCr Skin Colour Model for Human Face Detection - Nusirwan Anwar bin Abdul Rahman, Kit Chong Wei and John See <http://pesona.mmu.edu.my/~johnsee/research/papers/files/rgbhcbcr_m2usic06.pdf>  
[5] A. Conci, E Nunes, J. J. Pantrigo, A. Sanchez, “Comparing color and texture-based algorithms for human skin detection”, vol. 5, Computer Interaction pp. 168-173, 2008.  
[6] An Extended Set of Haar-like Features for Rapid Object Detection - Rainer Lienhart and Jochen Maydt <http://nichol.as/papers/Lienhart/An%20Extended%20Set%20of%20Haar-like%20Features%20for%20Rapid%20Object.pdf>
[7] OpenCV-Face Detection using Haar Cascades <http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html>  
