# DINOWaterNet
Water extraction in complicated scenarios.

## training dataset
We have built a large-scale water mapping dataset with ultra-high resolution optical remote sensing images, which can be seen below, and it can be downloaded from BaiduDisk, using the following link:
```java
WaterDataset
Link: https://pan.baidu.com/s/1oIvJrIIYzSgaAYuEuiFIAg passkey: 1234 
```

![sample images](img/samples.png)


## model structure
We use the iFormer-s as our backbone to build DINOWaterNet. The parameter and inference time can be tested by using the following code:
```python
python .\cal_Param.py
```
![model structure](img/model.png)

## performance test
We have tested the DINOWaterNet on very large-scale regions, and tested over 700GB ultra-high resolution remote sensing images, see below:
![model structure](img/large-scale-test.png)

## software development
Based on GDAL, wxPython, etc, we have developed a user-friendly software, which can run successfully in Windows 10/11/Server operating systems, and it can be downloaded from the following link:
```python
DINOWaterNet-Software
Link: https://pan.baidu.com/s/1LaaIqJ7IRvsRjrfoXe4jfA passkey: 1234 
```
![model structure](img/software.png)
