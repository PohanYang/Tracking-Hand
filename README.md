# Tracking Hand  
This is a project about tracking left hand and right hand for sign language estimation: <https://github.com/PohanYang/Sign-Language-Estimation-by-Tensorflow>  
  
  
Demo Video: <https://youtu.be/enSGG8Od8uM>  
Use Language: Python 2.7  
Use Tool: OpenCV  
  
It can tracking right hand and left hand, Marked right hand on RED and left hand on BLUE.  
It use skin detection algorithm [2] from webcam to segement skin. Figure 2 show the same image of figure 1 convert from RGB to HSV.  
HSV is the most common cylindrical-coordinate representations of points in an RGB color model, H for ???  
![fig1](https://github.com/PohanYang/Tracking-Hand/blob/master/img/orgpic.PNG)  
Figure 1. Original Picture.  
![fig1](https://github.com/PohanYang/Tracking-Hand/blob/master/img/hsv.PNG)  
Figure 2. Conver figure 1 fram RGB to HSV.  
  
Reference:  
[1] Hand tracking and gesture recognition system for human-computer interaction using low-cost hardware <https://link.springer.com/article/10.1007/s11042-013-1501-1>  
[2] Skin Detection using HSV color space - V. A. Oliveira, A. Conci  <http://webcache.googleusercontent.com/search?q=cache:http://www.matmidia.mat.puc-rio.br/sibgrapi/media/posters/59928.pdf&gws_rd=cr&ei=i87IWPa3LMuF8wWI05KgAw>  
[3] OpenCv Documentation  
[4] RGB-H-CbCr Skin Colour Model for Human Face Detection - Nusirwan Anwar bin Abdul Rahman, Kit Chong Wei and John See <http://pesona.mmu.edu.my/~johnsee/research/papers/files/rgbhcbcr_m2usic06.pdf>  
