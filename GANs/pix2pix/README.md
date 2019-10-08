﻿
# Pix2pix
*Image-to-Image Translation with Conditional Adversarial Networks*
#### 1. Authors

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros

[[Paper]](https://arxiv.org/abs/1611.07004)

#### 2. Introduction
영어를 프랑스어로 바꾸어 주는 것 처럼 한 이미지를 다른 이미지로 **변환(translating)** 해보자는 것이 이 논문의 핵심이다. 

![teaser_v3](https://phillipi.github.io/pix2pix/images/teaser_v3.png)
- 레이블 맵을 사진으로 변환
- edge maps에서 물체를 재구성
- 이미지 컬러화

위와 같은 결과를 내기 위해서 [Conditional GAN]([https://arxiv.org/abs/1411.1784](https://arxiv.org/abs/1411.1784))을 활용한다. 네트워크는 입력된 이미지에서 새로운 이미지를 만들어내며(generator), 생성된 이미지가 입력된 이미지의 pair 이미지와 같은지, 다른지(discriminator)를 통해 손실함수가 최적화되며 이미지 매핑을 배우게 된다. 

<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BcGAN%7D%28G%2C%20D%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%2Cy%7D%5BlogD%28x%2Cy%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%2Cz%7D%5Blog%281-D%28x%2C%20G%28x%2Cz%29%29%29%5D"\>
</p>
<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BL1%7D%28G%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%2Cy%2Cx%7D%5B%5Cleft%20%5C%7C%20y-G%28x%2Cz%29%20%5Cright%20%5C%7C_1%5D">
</p>
<p align="center">
    <img src="http://latex.codecogs.com/gif.latex?G%5E*%20%3D%20%5Carg%20%5Cmin_%7BG%7D%20%5Cmax_%7BD%7D%20%5Cmathcal%7BL%7D_%7BcGAN%7D%28G%2CD%29%20&plus;%20%5Clambda%20%5Cmathcal%7BL%7D_%7BL1%7D%28G%29">
</p>

>##### **수식기호 이해**
>- <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{x,y}[logD(x,y)]"/> \
 : <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{cGAN}(G,D)"/>의 첫번째 항으로 Discriminator에 이미지(x)와 정답 pair 이미지(y)가 input으로 들어간다. 
>- <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{x,z}[logD(1-D(x,G(x,z)))]"/> \
 : <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{cGAN}(G,D)"/>의 두번째 항으로 pair 이미지 중 하나가 Generator에 입력되며, Discriminator는 Generator가 생성한 이미지를 real or fake로 구분하게 된다.
>- <img src="http://latex.codecogs.com/gif.latex?%5Cmathbb%7BE%7D_%7Bx%2Cy%2Cz%7D%5B%5Cleft%20%5C%7C%20y-G%28x%2Cz%29%5Cright%20%5C%7C_1%5D"/> \
 : Reconstruction Loss로 Generator에 의해 생성된 이미지와 정답 이미지간 pixel 오차를 줄인다. 

<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{cGAN}(G,D)"/>는 [GAN](https://arxiv.org/abs/1406.2661)의 Loss와 같다. 하지만 noise를 입력하지 않고 pair 이미지 중 하나를 넣게 된다. 이것은 기존 GAN에서 할 수 없었던 **"원하는"** 이미지를 생성하기 위한 **조건부**로 활용된다. <img src="http://latex.codecogs.com/gif.latex?%5Cmathbb%7BE%7D_%7Bx%2Cy%2Cz%7D%5B%5Cleft%20%5C%7C%20y-G%28x%2Cz%29%5Cright%20%5C%7C_1%5D"/>는 Generator에서 생선된 이미지가 pair 이미지와 비슷함을 유지하기 위한 것이다.

더 많은 정보는 아래 참고 링크를 살펴보라.

#### 3. Network

 - Generator (U-net)
<p align="center">
    <img src="assets/pix2pix_unet.png">
</p>

 - Discriminator (Patch)
<p align="center">
    <img src="assets/pix2pix_patch.png">
</p>

#### 4. Implemenation

```
python pix2pix.py
```

#### 5. results (200 epoch)

<p align="center">
    <img src="assets/result_pix2pix.gif", width="360">
</p>

#### 참고
- [University of California, Berkeley In CVPR 2017](https://phillipi.github.io/pix2pix/)
- [Pix2Pix-라온피플 머신러닝 아카데이](http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=221356582945&categoryNo=22&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=1)
- [GAN을 이용한 Image to Image Translation: Pix2Pix, CycleGAN, DiscoGAN-Taeoh Kim's Blog](https://taeoh-kim.github.io/blog/gan%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-image-to-image-translation-pix2pix-cyclegan-discogan/)
