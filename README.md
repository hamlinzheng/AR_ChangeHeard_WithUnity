本项目通过PaddleHub的人脸关键点检测模型获取关键点坐标，并解算得到嘴巴、眼睛的动态数据以及头部的姿态数据，将数据发送到Unity内控制3d模型的动作同步，并实时将画面传回python端，将传回的头像p到实际人体即可实现AR换头功能。类似IOS里的Memoji、华为里的趣AR功能。

![](https://ai-studio-static-online.cdn.bcebos.com/f237dfce28bd4f5bbc820e4f9d55ce4698f61f1efccd499a97befd5179085d41)



本项目基于[OpenVHead](https://github.com/TianxingWu/OpenVHead)二次开发，有关Unity的使用可以参考原项目

