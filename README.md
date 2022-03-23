# OnnxSample

## Requirements
```
- Visual Studio >= Visual Studio 2015  
- .Net Framework >= 4.6.2
```

## Description
YOLOv5 및 segmentation_model.pytorch로 학습한 모델을 C#에서 사용하기 위한 샘플 코드

상세 설명은 아래 블로그 참고
- [C# 기반 배포 가능한 딥러닝 객체 감지 프로그램 개발 #1](https://medium.com/hbsmith/c-%EA%B8%B0%EB%B0%98-%EB%B0%B0%ED%8F%AC-%EA%B0%80%EB%8A%A5%ED%95%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B0%9D%EC%B2%B4-%EA%B0%90%EC%A7%80-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-%EA%B0%9C%EB%B0%9C-feat-yolo-v5-1-98581e397aa4)
- [C# 기반 배포 가능한 딥러닝 객체 감지 프로그램 개발 #2](https://medium.com/hbsmith/c-%EA%B8%B0%EB%B0%98-%EB%B0%B0%ED%8F%AC-%EA%B0%80%EB%8A%A5%ED%95%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B0%9D%EC%B2%B4-%EA%B0%90%EC%A7%80-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-%EA%B0%9C%EB%B0%9C-feat-yolo-v5-2-3310b8d81a82)
- [C# 기반 Semantic Segmentation 개발](https://minsu-cho.medium.com/c-%EA%B8%B0%EB%B0%98-semantic-segmentation-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-%EA%B0%9C%EB%B0%9C-d330a98a005b)


## Run
```
Usage: OnnxSample.exe [mode] (det or seg)
```
- det: Object  Detection 샘플 코드
- seg: Semantic Segmentation 샘플 코드