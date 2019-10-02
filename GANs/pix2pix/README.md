
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

(loss 함수 설명)

#### 3. Network

#### 4. Visualization

#### 5. Implemenation

#### 6. results (200 epoch)

<p align="center">
    <img src="assets/result_pi2pix.gif", width="360">
</p>

#### 참고
- [University of California, Berkeley  
In CVPR 2017]([https://phillipi.github.io/pix2pix/](https://phillipi.github.io/pix2pix/))
- [Pix2Pix-라온피플 머신러닝 아카데이]([http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=221356582945&categoryNo=22&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=1](http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=221356582945&categoryNo=22&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=1))
- [GAN을 이용한 Image to Image Translation: Pix2Pix, CycleGAN, DiscoGAN-Taeoh Kim's Blog](https://taeoh-kim.github.io/blog/gan%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-image-to-image-translation-pix2pix-cyclegan-discogan/)
