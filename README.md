# 🦴 Hand Bone Image Segmentation

## 1. 📖 프로젝트 소개

- 개요

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치는 기관으로 뼈를 정확히 인식하는 것은 의료 진단 및 치료 계획을 세우는데 필수적입니다. Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 딥러닝 기술을 활용한 뼈를 분할하는 것에 대한 많은 연구가 이뤄지고 있습니다. 

본 프로젝트는 사람 손의 X-Ray 촬영 이미지에 대해 Segmentation을 수행하는 것을 목적으로 하고 있습니다. 

<br/>

> 프로젝트 기간 : '24.11.11 ~ '24.11.28

<br/>

## 2.🧑‍🤝‍🧑 Team ( CV-20 : 수상한조)

<table>
    <tr height="160px">
        <td align="center" width="110px">
            <a href="https://github.com/IronNote"><img height="110px"  src="https://avatars.githubusercontent.com/IronNote"></a>
            <br/>
            <a href="https://github.com/IronNote"><strong>김명철</strong></a>
            <br />
        </td>
        <td align="center" width="110px">
            <a href="https://github.com/kaeh3403"><img height="110px"  src="https://avatars.githubusercontent.com/kaeh3403"></a>
            <br/>
            <a href="https://github.com/kaeh3403"><strong>김성규</strong></a>
            <br />
        </td>
        <td align="center" width="110px">
            <a href="https://github.com/kimmaru"><img height="110px"  src="https://avatars.githubusercontent.com/kimmaru"/></a>
            <br/>
            <a href="https://github.com/kimmaru"><strong>김성주</strong></a>
            <br />
        </td>
        <td align="center" width="110px">
            <a href="https://github.com/SuyoungPark11"><img height="110px" src="https://avatars.githubusercontent.com/SuyoungPark11"/></a>
            <br />
            <a href="https://github.com/SuyoungPark11"><strong>박수영</strong></a>
            <br />
        </td>
        <td align="center" width="110px">
            <a href="https://github.com/kocanory"><img height="110px" src="https://avatars.githubusercontent.com/kocanory"/></a>
            <br />
            <a href="https://github.com/kocanory"><strong>이승현</strong></a>
            <br />
        </td>
        <td align="center" width="110px">
            <a href="https://github.com/nOctaveLay"><img height="110px" src="https://avatars.githubusercontent.com/nOctaveLay"/></a>
            <br />
            <a href="https://github.com/nOctaveLay"><strong>임정아</strong></a>
            <br />
        </td>
</table> 

|Name        |Roles                                                         |
|:----------:|:------------------------------------------------------------:|
|김명철|  |
|김성규| EDA, 모델 실험(FCN), Offline Augmentation |
|김성주|  |
|박수영| 일정 관리, streamlit 구현, 모델(UNet계열, MANet, SegFormer) 실험, Hard voting |
|이승현| 베이스라인 수립, 모델 실험(DeepLab 계열, UperNet), Test Time Augmentation 수행, 하이퍼 파라미터 튜닝, |
|임정아| |

<br/>

## 3. 💻 프로젝트 수행 

### 3.1. 개발 환경 및 협업 도구

본 프로젝트를 진행한 환경 및 원활한 협업을 위해 사용했던 툴들은 다음과 같습니다.

- 서버      : V100 GPU
- 버전 관리 : Github
- 기록 관리 : Notion
- MLOps    : WandB 
- 기타      : Streamlit, Zoom

<br/>

### 3.2. 프로젝트 구조

프로젝트는 다음과 같은 구조로 구성되어 있습니다. 
```
📦level2-cv-semanticsegmentation-cv-20-lv3
 ┣ 📂config
 ┃ ┣ 📜config.yaml
 ┃ ┣ 📜soft_voting_config.yaml
 ┣ 📂data  # 별도 설치 필요 
 ┃ ┣ 📂test
 ┃ ┃ ┗ 📂DCM
 ┃ ┣ 📂test
 ┃ ┃ ┗ 📂DCM
 ┃ ┃ ┗ 📂output_json
 ┣ 📂eda
 ┃ ┣ 📜eda.ipynb
 ┃ ┣ 📜eda2.ipynb
 ┃ ┗ 📜run_streamlit.py
 ┣ 📂mmenv  # MMSegmentation 실행을 위한 별도 환경
 ┃ ┣ 📂config
 ┃ ┃ ┣ 📜config.py
 ┃ ┃ ┗ 📜pipeline.py
 ┃ ┣ 📂utils
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜metric.py
 ┃ ┃ ┣ 📜mixin.py
 ┃ ┃ ┗ 📜test.py
 ┃ ┗ 📜run.py
 ┣ 📂tools
 ┃ ┣ 📜custom_augments.py
 ┃ ┣ 📜custom_dataset.py
 ┃ ┣ 📜function.py
 ┃ ┗ 📜select_model.py
 ┣ 📜.gitignore
 ┣ 📜optimize.py
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┣ 📜run.py
 ┣ 📜test.py
 ┗ 📜train.py
```

### 3.3. 실행 방법

#### 3.3.1. yaml 파일 수정

config 폴더에 있는 yaml 파일의 경로, 모델 선정, 하이퍼 파라미터 등을 입력합니다.

<br/>

#### 3.3.2. 훈련 실행

3.3.1.에서 수정한 yaml 파일의 세팅에 따라 훈련을 진행하기 위해서는 run.py 파일을 실행해야 합니다. 해당 파일을 실행하기 위해서는 mode를 지정하고 config 파일 경로를 명시해야 하며, -m와 -c의 형태로 축약하여 사용 가능합니다.

```bash
$ python run.py --mode test --config ./config/config.yaml
```
<br/>

#### 3.3.3. 추론 실행

yaml 파일의 save_dir에 기재된 경로에 따라 저장된 모델(.pt)파일을 불러와 추론을 진행합니다. 훈련과 동일하게 run.py 파일의 mode만 test로 지정해 실행합니다. 

```bash
$ python run.py --mode test --config ./config/config.yaml
```
<br/>

#### 3.3.4. 하이퍼파라미터 최적화

yaml 파일에 명시된 모델에 알맞은 학습률, 옵티마이저, 스케줄러를 탐색합니다. 탐색한 파라미터는 yaml 파일의 save_dir에 기재된 경로에 json 파일로 저장됩니다. 훈련과 동일하게 run.py 파일의 mode만 opt로 지정해 실행합니다.

```bash
$ python run.py --mode opt --config ./config/config.yaml
```

앞서 언급한 방법은 Torchvision과 SMP 라이브러리에 있는 모델을 사용할 수 있는 방법입니다. 만약 MMSegmentation 라이브러리를 활용하기 위해서는 mmenv 폴더의 [README](https://github.com/boostcampaitech7/level2-cv-semanticsegmentation-cv-20-lv3/tree/main/mmenv)에서 확인할 수 있습니다.

<br/>

### 3.4. 수행 결과

#### 3.4.1. 모델별 성능 실험 결과

실험을 진행했던 모델별 수행 결과는 다음과 같습니다. (Dice 값은 Validation 평균 Dice 기준)

| Model    | Backbone       | Dice |
|----------|----------------|------|
|FCN       |ResNet50        |0.9374|
|UNet      |ResNet50        |0.9474|
|UNet++    |ResNet50        |0.9538|
|UNEt++    |ResNet101       |0.9517|
|UNEt++    |Efficientnet-b5 |0.9511|
|UNEt++    |GerNet-L        |0.9513|
|DeepLabV3 |ResNet101       |0.9420|
|DeepLabV3+|ResNet50        |0.9503|
|DeepLabV3+|ResNet101       |0.9495|
|DeepLabV3+|ResNext101_32x8d|0.9489|
|DeepLabV3+|Efficientnet-b8 |0.9515|
|DeepLabV3+|Xception71      |0.9512|
|UperNet   |ResNet101       |0.9479|
|UperNet   |swin transformer|0.9501|
|SegFormer |mit-b0          |0.9610|

(Default : 512 x 512, lr=1e-3, Optim=Adam, Epoch=100)

<br/>

#### 3.4.2. 모델 외 실험 결과

- 사이즈 변화에 따른 성능 비교

사이즈별 성능 비교를 위해 베이스라인인 FCN ResNet 50으로 Epoch을 낮춰 학습한 결과는 다음과 같습니다.

| Image Size     | Dice |
|----------------|------|
| 256 x 256      |0.8431|
| 512 x 512      |0.8575|
| 1024 x 1024    |0.9644|

- Offline Augmentation

상대적으로 낮은 Dice를 기록한 손가락 끝 및 손등 부분 뼈에 대한 이미지를 Crop하여 학습한 결과(값 : Dice)는 다음과 같습니다. 

| Class    | Original | 손목 Crop | 손가락 Crop | 손목 + 손가락 |
|----------|----------|-----------|------------|--------------|
|Lunate    |  0.9103  |  0.9366   |   0.9340   |    0.9316    |
|Trapezoid |  0.8705  |  0.8951   |   0.8917   |    0.9005    |
|Pisiform  |  0.8232  |  0.8750   |   0.8689   |    0.8864    |
|Trapezium |  0.9110  |  0.9221   |   0.9082   |    0.9178    |
|finger-16 |  0.8564  |  0.9060   |   0.9054   |    0.9002    |

<br/>

#### 3.4.3. 최적 훈련 조합 도출

개발 환경의 메모리 여건을 고려하여 아래와 같은 조합을 도출하였으며, Optuna를 통해 도출한 최적의 하이퍼파라미터를 적용하여 100 epochs 훈련을 진행했습니다. 

- Unet++ : Backbone(ResNet50), Image Size(1024)
- DeepLabV3 : Backbone(EfficientNet-b8), Image Size(1024)
- UperNet : Backbone(Swin Transformer), Image Size(1536)
- SegFormer : Backbone(mit-b0), Image Size(1024)

<br/>

#### 3.4.4. 최종 수행 결과

위 훈련 조합 결과, 단일 모델 중 가장 높은 성능을 기록한 것은 UperNet(Swin Transformer)였습니다. 다만, 모델 학습에 많은 시간이 소요된 관계로 Best가 없는 경우와 있는 경우로 나눠 앙상블을 진행하였으며, 그 결과는 아래와 같습니다. 

| Model               | Threshold | Dice |
|---------------------|-----------|------|
| Best (UperNet)      |     X     |0.9700|
| Ensemble (w/o Best) |     1     |0.9676|
| Ensemble (w/o Best) |     2     |0.9689|
| Ensemble (w/  Best) |     1     |0.9696|

보다 자세한 실험 과정 및 결과를 확인을 원하시는 분들은 [Wrap-up reaport](링크수정)를 참고하여 주시기 바랍니다.

<br/>
 
## 4. 기타사항

- 본 프로젝트에서 사용한 데이터셋은 부스트캠프 교육용 라이선스에 따라, 관련 이미지 및 데이터를 저장소 저장 또는 보고서 기재되지 않도록 처리하였습니다.
