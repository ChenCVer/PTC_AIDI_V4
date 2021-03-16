​		本代码是git的2020年6月28日的mmdetection代码，本分代码是从mmdet上修改而来的，mmdet的版本为：2.1.0，另外对应的mmcv代码版本为：0.6.1。

### 目前框架支持的模型：

#### classification：
- [x] ResNet
- [x] ResNeXt: https://zhuanlan.zhihu.com/p/78019001
- [x] SE-ResNet
- [x] SE-ResNeXt
- [x] ResNest: https://www.bilibili.com/video/BV1PV411k7ch#reply3078900535
- [x] Res2Net: https://zhuanlan.zhihu.com/p/86331579
- [x] RegNet: https://zhuanlan.zhihu.com/p/124091533
- [x] MobileNetV2: https://zhuanlan.zhihu.com/p/70703846
- [x] MobileNetV3: https://zhuanlan.zhihu.com/p/70703846
- [x] ShuffleNetV1: https://zhuanlan.zhihu.com/p/51566209
- [x] ShuffleNetV2: https://zhuanlan.zhihu.com/p/67009992
- [ ] RepVGG: https://zhuanlan.zhihu.com/p/344324470, https://zhuanlan.zhihu.com/p/352239591
- [ ] ACNet: https://zhuanlan.zhihu.com/p/104114196
- [ ] Efficientnet: https://zhuanlan.zhihu.com/p/67834114
- [ ] Inception
- [ ] Xception
- [ ] CSPNet
#### segmentation：
- [x] Unet
#### detection：
- [x] YOLOv3
- [x] YOLOv4
- [x] YOLOv5
- [ ] CenterNet (with Rotation)
- [ ] FCOS
- [ ] NanoDet
- [ ] CenterNetV2
#### keypoint：

## TODO: 

1. 语义分割添加注意力机制:《Concurrent Spatial and Channel "Squeeze & Excitation" in Fully Convolutional Networks》
2. yolov3和yolov4在voc数据集(多分类)上需要验证效果;
3. yolov5需要代码重构;
4. 支持.pth文件转换成.onnx中间文件格式, 并通过onnx-simlpifier对onnx进行去常量化操作，参考工程：
   https://github.com/daquexian/onnx-simplifier, 然后用TensorRT和ONNXRuntime的python接口进行模型验证和精度校对。
5. unet需要支持多尺度输入、多尺度输出(与yolov3代码风格一样)，残差连接(√)；
6. 分割训练支持datalayer自适应机制(adaptive adjust ok 等)；
7. 数据增强部分的代码有些需要重构，需要整合再一起；
8. rraitools文件夹和mmcv文件夹部分代码重复，需要合并整理；
9. 分类和分割的训练过程代码要与检测一致，也即不再走BaseHead类下的calc_train_batch_loss()这一部分，而是与检测一致, 
    尽量做到代码统一, 由于grad-cam这部分代码很难兼容，因此，分类的get_results部分暂时先不改动;
10. 分割测试支持整图被滑动窗口裁剪后小图测试，然后自动拼接成整图.(√)
11. 整理losses文件夹中的内容, 同时修改loss_wrapper.py下的GeneralLosser代码, 让其多尺度监督变成和mmdet的多尺度监督
    风格一模一样(√);
12. 本代码目前只完善了基于epoch变化学习率策略，还需要完善基于iter的策略;
13. 本份代码需要在win系统下验证可行性;
14. 参看: https://github.com/ChenCVer/segmentation_models.pytorch, 集成分割代码;
15. 参看: https://github.com/rwightman/pytorch-image-models, 集成分类代码;
16. 集成软分割领域的性能优异的网络U2-Net: https://github.com/NathanUA/U-2-Net;
17. 集成Unet++网络;
18. 对于yolo系列, 支持将PAN和BiFPN两种neck结构;
19. 集成移动端实用模型: Nanodet: https://github.com/RangiLyu/nanodet;
20. 集成目标检测领域的Generalized Focal Loss V1~2, 工业上很有用的loss;
21. 分类训练集成技巧: FixRes, 训练时采用较低分辨率(224*224),测试时采用较高分辨率;
22. 关于特征可视化, 代码不仅提供grad-cam，还提供集中cam, grad_cam++, score_cam等更好的可视化效果;
23. 参考LesionNet, 对输入层添加: CoordConv层, 给出坐标先验, 参考文章:
     https://arxiv.org/ftp/arxiv/papers/2012/2012.14249.pdf;
24. 添加回归框损失函数: EIOU Loss和带Focal-EIOU Loss;
25. 集成性能优异的关键点网络: CPN, HourglassNet等;
26. 集成性能优异, 速度快的repVGG分类网络(推理速度相当快,在1070显卡Res=224*224,
    inference时间不到1ms.);
27. 集成性能优异(工业上常用)的anchor-free, 目标检测算法: CenterNet, CenterNetV2, FCOS;
28. 集成关键点领域的优异网络: Hourglass, CPN;
29. 集成语义分割中常用loss, 参考: https://mp.weixin.qq.com/s/ra2qpFSbSuuJPDj39A5MWA;
30. 后期完善: mmdet2trt操作, 也即: .pth -> trt文件，参考: https://github.com/grimoire/mmdetection-to-tensorrt
31. 备注：后期进行模型部署(onnxruntime的c++接口)，针对onnx这块，参看：https://github.com/tenglike1997/onnxruntime-projects

## 知识系统化:
1. 通过张航博士讲解的ResNest设计理念，总结ResNet的网络变化的推演过程, 需要参考的知识如下:
   ①: https://zhuanlan.zhihu.com/p/66520078(√);

## 部署工程
   针对此项目训练出的模型，编写对应的部署工程，涉及ONNXRuntime和TensorRT等。
1. TensorRT系统讲解： https://edu.51cto.com/course/25834.html
2. Pytorch/onnx深度学习模型C++部署：https://edu.csdn.net/learn/30728
