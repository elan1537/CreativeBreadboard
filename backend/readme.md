# Backend

## 목적
Frontend 에서 전달받은 이미지 파일에서 components detection 과 detecting 된 components를 이용하여 전류와 전압의 수치, 회로도를 반환하는 API server. 

## 시작전에...
requirement.txt를 이용하여 의존성 해결. 아래의 코드 사용. 

```
pip install -r requirements.txt
```

## 시작하기
backend 폴더에서 다음의 명령어를 통해 시작. 

```
python server.py
```

## 파일 리스트

### server.py
Backend 구동을 위한 main 코드. 

### diagram.py
Detection 한 회로에 대해 이미지 파일로 변환. 

### (Depricated) findColor.py
전선 색을 detection 하기 위해 생성 **했던** 코드. 

### findComponents.py
컴포넌트 detection 을 위한 코드. 

### calcVoltageAndCurrent.py
Detection 된 components 에서 전류와 전압 계산. 

### (Depricated) warpedPinmap.csv


### (Depricated) warpedPinmap.json


## 폴더 리스트

### static
Backend 구동을 위한 기본 파일들. 

### (Depricated) templates
Jinja 사용 시 template를 담기위한 폴더. 