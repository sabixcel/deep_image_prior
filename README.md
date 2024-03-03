# deep_image_prior
Unsupervised image restoration

Official implementation of the research: "[Deep Image Prior](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf)”, by Dmitri Ulyanov, Andrea Vedaldi and Victor Lempitsky, published on CVPR in 2018.

Deep Image Prior (DIP) is a specific type of convolutional neural network (CNN) architecture that is used to improve the quality of an input image without relying on external or pre-trained data. Instead, it uses information contained in the input image itself as guiding "prior knowledge" to perform image enhancement or restoration.
Use cases in paper: image denoising, indoor painting, super resolution, jpeg artifact removal. But it can also be used for MRI reconstruction, surface reconstruction, pet reconstruction, audio denoising, time series problems, and more.

Following the implementation created by the authors of the original paper and also [Andrew Reader video](https://www.youtube.com/watch?v=FPzi8cUhNNY), I studied the case of image reconstruction: denoising + inpaiting.

![alt text](https://github.com/sabixcel/deep_image_prior/blob/main/figure6.png)

By running the code, the best NRMSE, 10.87%, was obtained at epoch 5583. After that, the normalized root mean square error increases, and the effect of overfitting was present. Therefore, the code must be stopped early to avoid overfitting. The best advantage of this implementation is that the model does not need data for training and learns all the information necessary to perform the task from the noisy input image. In the figure above the real image was used only for comparison, in real life we do not have access to it.
