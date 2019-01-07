

# GAN
*Generative Adversarial Network*
#### 1. Authors

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

[[Paper]](https://arxiv.org/abs/1406.2661)


#### 2. Introduction

>*" 내가 창조해내지 못하면, 나는 그것을 이해하지 못한 것이다."*<div style="text-align: right">- 리처드 파인만</div>


GAN을 설명할 때 꼭 나오는 리처드 파인만의 말이다. 이미지 분류에 탁월한 성능을 뽐내는 CNN은 새로운 이미지를 만들어내지 못한다. 그러나 GAN은 데이터를 이해한 듯 새로운 데이터를 만들어 낸다. 마치 파인만의 명언처럼 말이다.

GAN은 이름에서도 알 수 있드시 *'Adversarial' 적대적인* 두 개의 신경망이 경쟁하면서 성능을 향상되는 것이다. Ian Goodfellow는 '내쉬 균형(Nash Equilibrium)'이란 게임이론을 적용하여 GAN을 만들었다. 이처럼 놀라운 모델은 맥주를 마시다가 나온 아이디어란 것이다. 위대한 생각은 책상앞에서만 나오는 것이 아니니 잠깐 뇌를 식혀주면서 연구하자. 논문에서 *'위조지폐범(Generator)'* 과 *'경찰(Discriminator)'* 로 역할을 두어 이해를 돕는다. 

위조지폐(=가짜 데이터)를 생성하는 '생성자 (Generator)'와 위조지폐를 판별하는 '판별자 (Discriminator)'로 구성되어 있다. 생성자는 판별자를 속이기 위해 진짜 지폐같은 가짜 지폐를 만들어내고, 판별자는 교묘해지는 위조지폐를 판별하도록 학습이 진행된다. 이런 경쟁으로 생성자와 판별자의 능력이 발전하게 되고 결과적으로는 판별자가 진짜 지폐와 위조 지폐를 구별하기 힘든(구별할 확률 <img src="https://latex.codecogs.com/svg.latex?\Large&space;$P_d$"/>=0.5) 상황까지 가는 것이다.

참고 : [[Jaejun Yoo's Playground] - generative-adversarial-nets](http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html)

<p align="center">
<img src="image/new_gan_process.gif?raw=true" width="600px">
</p>

위 gif로 generator(지폐위조범)과 discriminator(경찰)이 어떻게 학습하는지 표현해보았다. 
1. 지폐위조범은 경찰도 속을만한 위조 지폐를 만들어낸다. 경찰은 진짜 지폐와 진짜(1)로 위조 지폐(0)을 가짜로 판별하도록 학습한다. 
2. 처음에 지폐위조범이 만든 위조지폐는 형편없기 때문에 쉽게 경찰은 위조인 것을 알아차린다. 지폐위조범은 조금씩 더 진짜 같은 위조지폐를 만들어 내면서 경찰을 시험한다. 
3. 그러다 경찰이 잘 못 판별하면 패널티(loss 값 상승)를 받게 된다. 위 gif에선 더 정교한 discriminator를 표현하기 위해 경찰이 교체된 것으로 표현하였지만, 조금 더 정교하게 판별하도록 학습된 것 뿐이다. 
4. 이 같은 일이 반복하다보면 지폐위조범은 정교한 위조 지폐를 만들게 된다.


#### 3. Formula Analysis
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\min_G&space;\max_D&space;V(D,G)&space;=&space;\mathbb{E}_{x\sim&space;p_{data}~(x)}[log&space;D(x)]&space;&plus;&space;\mathbb{E}_{z\sim&space;p_x(z)}[log(1-D(G(z)))]" title="\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{data}~(x)}[log D(x)] + \mathbb{E}_{z\sim p_x(z)}[log(1-D(G(z)))]" />
</p>

위의 수식은 D(discriminator), G(generator)의 min max 게임을 표현한 것이다. 언뜻 보면 매우 복잡하지만 하나씩 뜯어보면 간단히 이해할 수 있다. 우선 좌항의 식을 알아보자.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\min_G&space;\max_D&space;V(D,G)" title="\min_G \max_D V(D,G)" />
</p>

CNN에선 Loss function으로 표현하지만 GAN에선 Value function으로 표현한다. D는 Value를 **최대한** 많이 가지려고 노력하지만, G는 D가 Value를 많이 못가져가도록 **최소화**하게 한다는 것이다. 다음으로 우항을 알아보자.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\mathbb{E}_{x\sim&space;p_{data}(x)}[log&space;D(x)]&space;&plus;&space;\mathbb{E}_{z\sim&space;p_x(z)}[log(1-D(G(z)))]" title="\mathbb{E}_{x\sim p_{data}(x)}[log D(x)] + \mathbb{E}_{z\sim p_x(z)}[log(1-D(G(z)))]" />
</p>

>##### **수식기호 이해**
>- <img src="https://latex.codecogs.com/svg.latex?\Large&space;$\mathbb{E}$"/> = Expectiion(기댓값) = 평균
>- <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;x\sim&space;p_{data}~(x)" title="x\sim p_{data}~(x)" /> = 실제 데이터(진짜 지폐) 확률분포에서 x가 일어날 확률
>- <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;z\sim&space;p_x(z)" title="z\sim p_x(z)" /> = 노이즈 데이터 확률분포에서 z가 일어날 확률
>- <img src="https://latex.codecogs.com/svg.latex?\Large&space;$D(x)$"/> = Discriminator에 실제 데이터를 넣었을 때 ouput(0과 1사이의 값)
>- <img src="https://latex.codecogs.com/svg.latex?\Large&space;$G(z)$"/> = Generator에 노이즈 데이터를 넣었을 때 ouput (가짜 그림)
>- <img src="https://latex.codecogs.com/svg.latex?\Large&space;$D(G(x))$"/> = 가짜 그림을 Discriminator에 넣었을 때 output(0과 1사이의 값)

##### 3.1 Discriminator
![from_the_view_of_the_discriminator](https://trello-attachments.s3.amazonaws.com/5bdbdb69221aa411080cfc37/5c2ca143b87c81663db19b92/d47a3aeaa48f6cf6310fbbed4f393878/gan2.png)

+를 기준으로 좌측 식을 보자. <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\mathbb{E}_{x\sim&space;p_{data}~(x)}[log&space;D(x)]" title="\mathbb{E}_{x\sim p_{data}~(x)}[log D(x)]" /> 이 식은 D가 진짜 지폐를 보았을 때 진짜 지폐라고 판별하도록 학습하는 수식이다. 반면에 우측 식 <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\mathbb{E}_{z\sim&space;p_x(z)}[log(1-D(G(z)))]" title="\mathbb{E}_{z\sim p_x(z)}[log(1-D(G(z)))]" />은 G가 만들어낸 가짜 지폐를 가짜로 판별하도록 학습하는 식이다. 

왜 그렇게 되는지 log(x), log(1-x)에 대한 그림을 살펴보자.

log(x) | log(1-x)
--- | ---
![log(x)](https://trello-attachments.s3.amazonaws.com/5bdbdb69221aa411080cfc37/5c2ca143b87c81663db19b92/3374ff4dc402b5ca5ca0da88232e0fcf/log(x).png) | ![log(1-x)](https://trello-attachments.s3.amazonaws.com/5bdbdb69221aa411080cfc37/5c2ca143b87c81663db19b92/c127c2ad52066fc570b3d486af509c38/log(1-x).png)

진짜 지폐(<img src="https://latex.codecogs.com/svg.latex?\Large&space;$x$"/>)는 D가 진짜(1)로 판단해야 하므로 <img src="https://latex.codecogs.com/svg.latex?\Large&space;$log(x)$"/>그래프에서 V(D,G)값이 증가하는 것을 알수 있다. (범위 : <img src="https://latex.codecogs.com/svg.latex?\Large&space;$-\infty$"/> ~ 0) 반대로 <img src="https://latex.codecogs.com/svg.latex?\Large&space;$log(1-x)$"/>그래프에선 가짜 이미지(<img src="https://latex.codecogs.com/svg.latex?\Large&space;$D(G(z))$"/>)를 가짜로 판단해야 하므로 그래프를 따라 V(D,G)값이 증가한다. GAN 소개글에서 D는 V(D,G)값을 최대한 증가시키려하는 것과 일치한다.


##### 3.2 Generator
![from_the_view_of_the_generator](https://trello-attachments.s3.amazonaws.com/5bdbdb69221aa411080cfc37/5c2ca143b87c81663db19b92/852e52f7d07f6a907932d68dc2c6d648/gan3.png)

G의 입장에서 보면 <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\mathbb{E}_{x\sim&space;p_{data}~(x)}[log&space;D(x)]" title="\mathbb{E}_{x\sim p_{data}~(x)}[log D(x)]" />은 D에 관한 것으로 무시할 수 있다. 우리는 우측에 있는  <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\mathbb{E}_{z\sim&space;p_x(z)}[log(1-D(G(z)))]" title="\mathbb{E}_{z\sim p_x(z)}[log(1-D(G(z)))]" />만 고려하면 된다. G는 D와 정반대로 V(D,G)값을 최소화하려 하기 때문에 D가 자신이 만든 가짜 지폐를 진짜 지폐로 오인하도록 만들어야 한다. 이 처럼 서로다른 목적을 가진 두 그룹은 value값을 두고 끝임없이 경쟁하면서 성장하게 된다.


위의 두 식을 아래처럼 도식화할 수 있다.
![gan_formula1](https://trello-attachments.s3.amazonaws.com/5bdbdb69221aa411080cfc37/5c2ca143b87c81663db19b92/3f65251a4179887aa7ea2cbba79cfc8c/gan1.png)


#### 4. Implementation

    python gan.py


#### 5. Result
![result](https://trello-attachments.s3.amazonaws.com/5bdbdb69221aa411080cfc37/5c2ca143b87c81663db19b92/6bf3f6b9e80e3b46f664c7564c16ef6a/gan.gif)
