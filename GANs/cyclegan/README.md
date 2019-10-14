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
<p align='center'>
    <img src="https://camo.githubusercontent.com/45a9327ea9e8a917a1f9d84312038a50966417c0/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6d61746863616c7b4c7d5f7b47414e7d28472c445f592c582c5929203d205c6d61746862627b457d5f7b79205c73696d20705f646174612879297d5b6c6f6744792879295d202b205c6d61746862627b457d5f7b78205c73696d20705f646174612878297d5b6c6f6728312d445f59284728782929295d">
</p>


<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?\mathcal{L}_{cyc}(G,F) = \mathbb{E}_{x \sim p_data(x)}[\left \| F(G(x)) - x \right \|_1] + \mathbb{E}_{y \sim p_data(y)}[\left \| G(F(y)) - y \right \|_1]"\>
</p>
<p align="center">
    <img src="https://camo.githubusercontent.com/5c0fbbf488b8f604a5316a6006f3a4b1dfa1c247/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6d61746863616c7b4c7d5f7b6379637d28472c4629203d205c6d61746862627b457d5f7b78205c73696d20705f646174612878297d5b5c6c656674205c7c2046284728782929202d2078205c7269676874205c7c5f315d202b205c6d61746862627b457d5f7b79205c73696d20705f646174612879297d5b5c6c656674205c7c2047284628792929202d2079205c7269676874205c7c5f315d">
</p>


<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?\mathcal{L}(G,F,D_X,D_Y) = \mathcal{GAN}(G,D_Y,X,Y) + \mathcal{GAN}(F,D_X,Y,X) + \lambda \mathcal{cyc}(G,F)"\>
</p>
<p align="center">
    <img src="https://camo.githubusercontent.com/bcd32f596e70d0869400cf7006e896c0d55590b8/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c6d61746863616c7b4c7d28472c462c445f582c445f5929203d205c6d61746863616c7b47414e7d28472c445f592c582c5929202b205c6d61746863616c7b47414e7d28462c445f582c592c5829202b205c6c616d626461205c6d61746863616c7b6379637d28472c4629">
</p>


<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?G^*, F^* = arg min_(G,F) max_(D_x,D_Y) \mathcal{L}(G,F,D_X,D_Y)"\>
</p>
<p align="center">
    <img src="https://camo.githubusercontent.com/a6c3db2d3b4dacdb226da18c1ec175e24dae05b0/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f475e2a2c20465e2a203d20617267206d696e5f28472c4629206d61785f28445f782c445f5929205c6d61746863616c7b4c7d28472c462c445f582c445f5929"\>
</p>


> **tricks**
> - *GAN Loss 함수 변경*  
CycleGAN에서 cross-entropy를 사용하면 gradeint vanishing 문제가 발생한다.
실험에서 LSGAN(Least Square GAN)으로 변경하면 더 좋은 성능이 나와서 사용하였다고 한다.
> - *L1 loss*  
말을 얼룩말로 바꾸는 네트워크 F와 그 반대로 얼룩말을 말로 바꾸는 네트워크 G를 이용하여 L1 loss를 적용한다.
말 정답이미지(y)를 F()에 넣어 얼룩말을 생성하고, 생성된 얼룩말을 G()에 넣어 다시 말로 변환한 것을 정답이미지 y와 pixel 비교한다.
G(), F() 네트워크가 Input 특성을 너무 많이 바꾸지 않도록 하게 한다.
> - *Identity mapping loss*  
또한, G() 네트워크에 정답 이미지(y)를 넣어서 만들어진 생성물과 정답 이미지의 L1 loss를 사용하여 
조금 더 사실적인 사진 이미지로 변환하는데 도움을 준다. 


**CycleGAN의 한계**
- 모양을 바꾸기가 어려움
- 데이터 분포

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
