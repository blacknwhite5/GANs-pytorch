# DCGAN
*Unsupervised Representation Learning with **D**eep **C**onvolutional **G**enerative **A**dversarial **N**etworks*
#### 1. Authors

Alec, Radford, Luke Metz, Soumith Chinatala

[[Paper]](https://arxiv.org/abs/1511.06434)

#### 2. Introduction

 1) 기존 GAN의 한계
  - 기존의 [GAN(Generative Adversarial Network)](https://arxiv.org/abs/1406.2661)은 결과가 **불안정**하다
  - Neural Network의 한계로 결과가 나오게 되는 과정이 **Black-box**이다.
  - Generative Model이 생성한 Samples이 기존 Sample과 비교하여 얼마나 비슷한지 확인할 수 있는 척도가 없다. (**주관적 기준**)

 2) DCGAN이 기여한 점
  - 대부분의 상황에서 언제나 **안정적인 학습구조**를 제안
  - **벡터산술(Vector arithmetic) 연산**이 가능
  - 학습한 filter를 시각화, filter들이 이미지의 **특정 물체를 학습했다는 것**을 보여줌
  - 학습된 Discriminator가 다른 비지도 학습 알고리즘과 비교하여 비등한 이미지 분류 성능을 보여줌

 3) DCGAN 학습 측정기준
  > 1. Generator가 **이미지를 외워서 보여주는 것이 아니란 것**
  > 2. **latent space(z)가 움직일 때 급작스런 변화가 일어나는 것이 아니라 부드러운 변화를 보여주는 것**


#### 3. Network
- Generator & Discriminator

![dcgan](https://trello-attachments.s3.amazonaws.com/5cc68d6179360c5cf3faa8c0/1000x233/d836da1ab28bd63790310fbf431ebaef/dcgan.png)

- fractional-strided convolutions vs CNN

convolution | fractional-strided
--- | ---
![conv](https://trello-attachments.s3.amazonaws.com/5cc68d6179360c5cf3faa8c0/395x381/d5dc0d8b388ef5a01ab15e6be342c68a/5938442c6336f86c21e8d6d3d59e2a98.gif) | ![fractional](https://trello-attachments.s3.amazonaws.com/5cc68d6179360c5cf3faa8c0/395x449/2a5534d6a537b12db5251f92acafac96/fe1b314e8e7c79f25d559ed0a1676aa1.gif)


gif 출처) https://github.com/vdumoulin/conv_arithmetic

#### 4. Visualization
 1) Walking in the latent space
    - latent space(z)를 조금씩 변화시키면 생성된 이미지가 급격히 변하는 것이 아니라 서서히 변화한다.
    ![walking-in-the-latent-space](https://trello-attachments.s3.amazonaws.com/5cc68d6179360c5cf3faa8c0/789x157/109f7564c12bb6a8dcaa3bcc2c19d947/image.png)

 2) Visualizing the discriminator features
    - Discriminator의 filter가 특정 물체를 학습했다는 것을 보임
    ![filter](https://trello-attachments.s3.amazonaws.com/5cc68d6179360c5cf3faa8c0/473x238/cfc48a1025eeab070b1ab73780592efb/image.png)

 3) Vector arithmetic
    - (선글라스를 쓴 남자) **-** (선글라스를 안쓴  남자) **+** (선글라스를 안쓴 여자) **=** (안경을 쓴 여자)
    - 각 이미지를 생성하는 latent space(z)를 산술적 연산으로 새로운 데이터를 생성할 수 있다.
    ![vector-arithmetic](https://trello-attachments.s3.amazonaws.com/5cc68d6179360c5cf3faa8c0/786x403/b221b8ad45205d25f54a2b1d01a5a914/image.png)


#### 5. Implemenation

 - Download LSUN dataset
```
python download.py -c bedroom
```

 - Place LSUN dataset in ```GANS-pytorch/data/```
 - Training


```
python dcgan.py
```

#### 6. results (10 epoch)

 - celebA  
 ![celebA](https://trello-attachments.s3.amazonaws.com/5cc68d6179360c5cf3faa8c0/332x332/a1d1069ed5dff1784782918361f98792/dcgan_celeba.gif)

 - LSUN  
 ![lsun](https://trello-attachments.s3.amazonaws.com/5cc68d6179360c5cf3faa8c0/332x332/2d1e0e5f9c28b09728f37cbcab16b280/dcgan_lsun.gif)



#### 참고
- [초짜 대학원생의 입장에서 이해하는 Deep Convolutional Generative Adversarial Network (DCGAN)](http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-1.html)
- [angrypark DCGAN 논문 이해하기](https://angrypark.github.io/generative%20models/paper%20review/DCGAN-paper-reading/)
