# deep_image_prior
Unsupervised image restoration 

Official implementation of the paper "[Deep Image Prior]([url](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf)https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf)", by Dmitry Ulyanov, Andrea Vedaldi and Victor Lempitsky, published on CVPR in 2018.

Deep Image Prior (DIP) is a specific type of convolutional neural network (CNN) architecture that is employed to improve the quality of an input image without relying on any external or pre-trained data. Instead, it utilizes the information contained within the input image itself as the guiding "prior knowledge" to perform image enhancement or restoration.
Use cases in the paper: image denoising, inpainting, super-resolution, jpeg artifacts removal. But it can also be used for MRI reconstruction, surface reconstruction, pet reconstruction, audio denoising, time series problems and many more.

Following the implementation created by the authors of the original paper and also [Andrew Reader video]([url](https://www.youtube.com/watch?v=FPzi8cUhNNY)https://www.youtube.com/watch?v=FPzi8cUhNNY), I studied the case of image reconstruicion: denoising + inpaiting.  

