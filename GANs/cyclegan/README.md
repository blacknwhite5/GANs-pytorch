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

하지만 기존의 GAN loss를 사용하면 문제점은 입력으로 넣은 그림에서 target 이미지로 생성되는 것을 막을 수 없다. 즉, 모네 그림을 넣어서 사진같은 모네 그림을 얻는 **스타일**을 배우는 것이 아니라 target 이미지와 비슷한 생성물을 얻게될 수 있다.

CycleGAN의 loss 컨셉은 입력 그림에서 생성된 이미지가 다시 입력 그림으로 재생성이 될 수 있게 강제하는 것이다. 스타일을 바꾸지만 원래 그림으로 복구가능할 정도만 바꾸게 제약을 거는 것이다. 이 제약을 걸어주는 loss는 pix2pix의 pixel level difference를 추가해주는 것과 같다.

#### 3. Network

#### 4. Implemenation
```
python cyclegan.py
```

#### 5. Visualization

#### 6. results (200 epoch)

#### 참고
- [junyanz.github.io](https://junyanz.github.io/CycleGAN/)
- [GAN을 이용한 Image to Image Translation: Pix2Pix, CycleGAN, DiscoGAN-Taeoh Kim's Blog](https://taeoh-kim.github.io/blog/gan%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-image-to-image-translation-pix2pix-cyclegan-discogan/)
