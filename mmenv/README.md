## How to set up 
- MMSegmentation 설치 ( 참고 : [Link](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation) )
```
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```


## How to implement

실제로 적용 하기 위해서는 run.py 파일을 실행해야 합니다. 해당 파일을 실행하기 위해서는 mode와 config 파일 경로를 명시해야 하며, -m와 -c의 형태로 축약하여 사용 가능합니다.

### Train
```bash
# Exmaple 
python run.py --mode train --config ./config/config.py
```

### Test 
```bash
# Exmaple
python run.py --mode test --config ./config/config.py
```
