# CycleGAN
*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*

#### 1. Authors

Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

[[Paper]](https://arxiv.org/abs/1703.10593)

#### 2. Introduction
pix2pix는 흑백과 컬러사진, 건물과 레이블맵 등 pair한 이미지가 사용되었다. 
그러나 현실세계에서는 이러한 데이터를 구성하는게 쉬운일은 아니다. 
가령 여름사진을 겨울사진으로 바꾸려는 모델을 만들려고 한다. 
그러면 같은 위치, 같은 구도에서 여름, 겨울에 촬영한 사진이 필요하다. 
누가 이렇게 데이터를 모으려 할까? 
CycleGAN은 pair하지 않은 이미지들로 학습하여 서로의 스타일로 바꿔주는 것을 할 수 있다.

<p align="center">
    <img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg">
</p>

하지만 **기존의 GAN loss를 사용하면** 문제점은 입력으로 넣은 그림에서 target 이미지로 생성되는 것을 막을 수 없다. 
즉, 모네 그림을 넣어서 사진같은 모네 그림을 얻는 **스타일**을 배우는 것이 아니라 target 이미지와 비슷한 생성물을 얻게될 수 있다.
- Input의 특성이 무시될 수 있음
- 같은 Output으로 매몰될 가능성 있음

**CycleGAN의 loss 컨셉**
입력 그림에서 생성된 이미지가 다시 입력 그림으로 재생성이 될 수 있게 강제하는 것이다. 
스타일을 바꾸지만 원래 그림으로 복구가능할 정도만 바꾸게 제약을 거는 것이다. 
이 제약을 걸어주는 loss는 pix2pix의 pixel level difference를 추가해주는 것과 같다.


<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?\mathcal{L}_{GAN}(G,D_Y,X,Y) = \mathbb{E}_{y \sim p_data(y)}[logDy(y)] + \mathbb{E}_{x \sim p_data(x)}[log(1-D_Y(G(x)))]"\>
</p>

<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?\mathcal{L}_{cyc}(G,F) = \mathbb{E}_{x \sim p_data(x)}[\left \| F(G(x)) - x \right \|_1] + \mathbb{E}_{y \sim p_data(y)}[\left \| G(F(y)) - y \right \|_1]"\>
</p>

<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?\mathcal{L}(G,F,D_X,D_Y) = \mathcal{GAN}(G,D_Y,X,Y) + \mathcal{GAN}(F,D_X,Y,X) + \lambda \mathcal{cyc}(G,F)"\>
</p>

<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?G^*, F^* = arg min_(G,F) max_(D_x,D_Y) \mathcal{L}(G,F,D_X,D_Y)"\>
</p>


<p align="center">
    <img src="https://trello-attachments.s3.amazonaws.com/5d663bdd41c39a1ad9983fff/1200x674/61a5de2aab22d1868a8b7ac7b0e24b72/image.png", width=720>
</p>


**Tricks**
- *LSGAN loss로 변경*  
CycleGAN에서 cross-entropy를 사용하면 gradeint vanishing 문제가 발생한다.
실험에서 LSGAN(Least Square GAN)으로 변경하면 더 좋은 성능이 나와서 사용하였다고 한다.

<p align="center">
    <img src="https://trello-attachments.s3.amazonaws.com/5d663bdd41c39a1ad9983fff/1200x675/bd6f322cae23a40ec1eecd5c41ad4644/image.png", width=720>
</p>

- *L1 loss*  
말을 얼룩말로 바꾸는 네트워크 F와 그 반대로 얼룩말을 말로 바꾸는 네트워크 G를 이용하여 L1 loss를 적용한다.
말 정답이미지(y)를 F()에 넣어 얼룩말을 생성하고, 생성된 얼룩말을 G()에 넣어 다시 말로 변환한 것을 정답이미지 y와 pixel 비교한다.
G(), F() 네트워크가 Input 특성을 너무 많이 바꾸지 않도록 하게 한다.

- *Identity mapping loss*  
또한, G() 네트워크에 정답 이미지(y)를 넣어서 만들어진 생성물과 정답 이미지의 L1 loss를 사용하여 
조금 더 사실적인 사진 이미지로 변환하는데 도움을 준다. 


**CycleGAN의 한계**
- *모양(형태)을 바꾸기가 어려움*  
사과를 오렌지로 생성할 때, 색만 변하고 형태는 사과모양을 유지

<p align="center">
    <img src="https://trello-attachments.s3.amazonaws.com/5d663bdd41c39a1ad9983fff/1200x676/3af08a8603755149c2e359e69356b7fd/image.png", width=720>
</p>


- *데이터 분포*  
사람을 태운 말을 얼룩말로 바꿀때, 사람까지 얼룩무늬로 만드는 현상. 
학습데이터에서 사람을 태운 말의 이미지가 없어서 발생한 것으로 보임

<p align="center">
    <img src="https://trello-attachments.s3.amazonaws.com/5d663bdd41c39a1ad9983fff/1200x675/8a9f3d4ab13e0d2a6b04972d8d68b7c6/image.png", width=720>
</p>

#### 3. Network


#### 4. Implemenation
```
python cyclegan.py
```

#### 5. Visualization

<p align="center">
    <img src="assets/visualize.png", width="360">
</p>

#### 6. results (200 epoch)


#### 7. TODO
- [ ] make replay buffer

#### 참고
- [junyanz.github.io](https://junyanz.github.io/CycleGAN/)
- [GAN을 이용한 Image to Image Translation: Pix2Pix, CycleGAN, DiscoGAN-Taeoh Kim's Blog](https://taeoh-kim.github.io/blog/gan%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-image-to-image-translation-pix2pix-cyclegan-discogan/)
- [Finding connections among images using CycleGAN](https://www.slideshare.net/NaverEngineering/finding-connections-among-images-using-cyclegan)
- [CycleGAN이 무엇인지 알아보자](https://www.slideshare.net/JackLee27/cyclegan-110117260)
