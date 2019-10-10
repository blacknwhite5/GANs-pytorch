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

#### 3. Network

#### 4. Implemenation
```
python cyclegan.py
```

#### 5. Visualization

#### 6. results (200 epoch)

#### 참고
- [junyanz.github.io](https://junyanz.github.io/CycleGAN/)
