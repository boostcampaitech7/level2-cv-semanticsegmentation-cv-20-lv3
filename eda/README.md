## EDA 설명

1. streamlit을 활용하여 Train, Test, Inference한 데이터셋을 시각화하는 것을 목적으로 합니다.

### streamlit 사용 방법
1. csv 파일 준비하기 : 추론된 결과를 확인하고 싶은 경우 RLE로 인코딩된 csv 파일을 eda 폴더로 이동 

2. run_streamlit.py 실행하기 : eda로 워킹 디렉토리를 변경하고 run_streamlit.py 실행 
```
cd eda
streamlit run run_streamlit.py
```

3. 확인하고 싶은 이미지에 따라 모드(Train/Inferred Train/Test)를 선택
- Train : 이미지와 Ground Truth 간 비교
- Inferred Train : 추론된 Train 데이터셋 결과와 Ground Truth 간 비교. 파일명(.csv 포함) 입력 필요
- Test : Test 이미지와 추론된 Test 데이터셋 결과 간 비교. 파일명(.csv 포함) 입력 필요 