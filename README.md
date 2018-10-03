# pytorch-CapsNet
Moduled CapsNet implemented by Pytorch

This project is based on the paper [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf) 
but the network is designed to tackle with Cifar-10 dataset.

In the network's architecture I use two DigitsCaps Layer which is renamed to RouteLayer so the test accuracy is around 10% that is consistent with the result of the paper [Capsule Network Performance on Complex Data](https://arxiv.org/pdf/1712.03480.pdf). Remove one of the RouteLayer and tune some params then the result should be around 60%.
Actually I finished my work first then I found the second paper...

Any way, this is my very first complete Pytorch project.
