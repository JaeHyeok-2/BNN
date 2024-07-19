# Binary_Deepfake_detection Convert TorchLightning To Torch

## 바꿔야 할 항목들 

- [x] Model.py, train.py에 대해서 Torch로 변환 

- [ ] Rendering의 결과에 대해서도 적용할 수 있도록 (One_image) dataloader or Training 함수에 대해서 이미지 변경 



## 실행
- 기존의 방식과 동일하게 진행, lego dataset을 기반으로 진행 
- train.py Line 76에서 cifake 대신 lego로 이름만 변경하여 진행하였음. 

```python train.py -cfg configs/results_lego_T_unfrozen.cfg```
