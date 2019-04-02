# VFX HW1
> b05902019 資工三 蔡青邑 
> b05902040 資工三 宋易軒
## Introduction 
This is our report for homework1 (NTUCSIE VFX 2019). And the code here shows how we implemented ==HDR image==, along with ==image adjustment== and ==tone mapping==.<br>
At the end, we ran our code on two sets of images (`View1` and `View2` in `testing_images`). And finally choose the result of `View1` as our final submision.<br>
In this report we'll show we weimplement images.
## How to reproduce our code
simply
```python=
python3 HDR.py
```
You're whelcomed to change the input path of `readImages()` in our main function to change the images (depends on which images set you'd like to implement on). 

## MTB alignment
We've implemented the MTB(Median Threshold Bitmap) alignment in function `MLT_alignment(images, template)`. 
- Firstly, compute the grey value of each images by ==54 * r + 183 * g + 19 * b==, and binarize the images by the median value(the threshold bitmap). 
- Secondly, compute the exclusive bitmap by change the pixel value which is too close (< 4) to threshold to 0.
- Thirddly, compute the errors compared with the template image, and then record the least-error-shift. 
- Lastly, by recursively divide the image in half, and repeat the above step,we obtain the fianl aligned images with the shitfed direction.



| R | G | B |
| -------- | -------- | -------- |
| ![](https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/reds.png?token=AVgtIqVI8LsEUWubLpvxpHIxYHw-ZYfOks5crKkBwA%3D%3D)     | ![](https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/greens.png?token=AVgtIp0-GmFJ7Z70C0q_p98yqn8ABZBBks5crKiuwA%3D%3D)     | ![](https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/blues.png?token=AVgtIqDJfLC-gCYEh7NzMx62_obmmrJ8ks5crKh4wA%3D%3D)     |



| Grey | Bitmap |
| -------- | -------- |
| ![](https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/grey.png?token=AVgtIt5z3EGieBEAhE7mjKNa77oEcTiMks5crKlbwA%3D%3D)    | ![](https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/tb.png?token=AVgtIqUfGMv6BwYlmgar-eATngjAe019ks5crKl_wA%3D%3D)   |



## Sampling image
The original images consists of 9 images with different shutter times. To access them in a single matrix we have to sample the intensity value on the same position for each value in our intensity range(0~255). That is, we create an array of the size of 256 * 9, and by checking where each value appears on our median image, accordingly search for the value of same position on other images.


## Response Curve
### Intensity weighting

We use a weight function to calculate a weight for each pixel, which can reduce the bias from extreme intensity when calculate the response fuction. 
```python3
def intensity_weighting(pix_value, MAX_intensity):
    return(min(pix_value, MAX_intensity - pix_value))
```

### Calculation
Minimize the following optimization equation to compute response function.
<center><img src="https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/response_curve.png?token=AVgtItOv2hJXcamaA2at9UcYfexZd0KVks5crRUTwA%3D%3D" width = 205 heigth = auto/></center>
Sample different pixels with same intensity from resource images, construct the corresponding matrix mentioned in the lecture slide and calculate the inverse matrix. Then we could recover response function.
The figure below shows $ln E$ with respect to intensity ranged from 0 to 255.

<center><img src="https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/RCurve1.png?token=AVgtIkM4v6-7bY6RyYOIkGRFNu5E0LiEks5crKqLwA%3D%3D" width = 305 height = auto/></center>

## Radiance Map
After we construct the response curve, we combine pixels to reduce noise and obtain a more reliable estimation along with the weight function according to our lecture.
(While implementing, we have to deal with some exception shuch that the sum of weight may be 0)

<center><img src="https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/RmapEquation.png?token=AVgtIlSWEsONnvXQTkubwLAfpcpjGR4-ks5crKv6wA%3D%3D" width=205 height=auto/></center>

```python
if SumW > 0:
    rad_map[i][j] = np.sum(w * (g - shutter_times) / SumW)
else:
    idx = rd.randrange(len(images))
    rad_map[i][j] = g[idx] - shutter_times[idx]
```
Then we draw the radiance map.

<center><img src="https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/results/RadianceMap1.png?token=AVgtIuy0MjfhLdCT7qO1QQWHHng_du55ks5crRPhwA%3D%3D" width=505 height=auto/></center>

 <div style="page-break-after: always;"></div> 

## Bilateral Tone Mapping

We implement the bilateral tone mapping, using a 5*5 Bilateral Filter with Gaussian fuction for range and spatial kernel, which can separate detail layer and base layer of the image, and reduce the contrast of base layer. Then, combine them and recover the image. By the method, we could preserve the detail in overexposured and dark area.
Our implementation is in appendix

### Bilateral Filter Definition
<center><img src="https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/Bilateral_Definition.png?token=AVgtIgImEq809yZ02Wzm7ouWn-_2Vf0fks5crRVAwA%3D%3D" width = 205 height = auto></center>

### Normalize Term
<center><img src="https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/Normalization_Term.png?token=AVgtImi0uD6IVb_NdVkdTLCeTGIupdnPks5crRVpwA%3D%3D" width=205 height=auto/></center>

### Procedure

1. Compute intensity by 0.2126 * R + 0.7152 * G + 0.0722 * B
2. Divide each color channel by intensity to obtain color layer
3. Apply Bilateral Filter on log intensity to get base layer
4. Subtract intensity by base layer to get detail layer
5. Compress base layer with specific factor
6. Add detail layer to adjusted base layer
7. Recover the log intensity by exponential
8. Multiply each color channel by adjusted intensity

### Experiement
Use different compression factor to adjust the magnitude of detail and contrast of base.


|0.5 |0.6 |0.7|
|-|-|-|
|![](https://i.imgur.com/JbkVvVC.png)|![](https://i.imgur.com/MMo6ZZm.png)|![](https://i.imgur.com/g5R7dDB.png)|
|0.8 |0.9 | 1.0|
|![](https://i.imgur.com/oG4S4ZC.png)|![](https://i.imgur.com/JHjzQ3G.png)|![](https://i.imgur.com/7iGFplg.png)|

After experiement, we decide to use 0.85 as our factor.

<center><img src="https://i.imgur.com/8GgnikX.png" width = 405 height = auto/></center>

## Gamma tone mapping
Also, we implement the gamma tone mapping, which is rather easy by 
```python
def gammaToneMapping(image):
    gamma = 0.8
    image_corrected = cv2.pow(image/255., 1.0/gamma)
    return image_corrected
```
## Result
The final result is at `./results/Result1.png`.
Before that, we compare differnt result by applying different tone mapping below, and choose the current best one.

- Gamma + Bilateral
- Gamma
- Bilateral
- Nothing(only normalize)


| Gamma + Bilateral | Gamma |
| -------- | -------- |
| ![](https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/biliteral_gamma.png?token=AVgtInHzC7fObEUThAKHSHIfF8_ZyYr_ks5crLDVwA%3D%3D)    | ![](https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/gamma.png?token=AVgtIgqs0YQYFfLVaGmvz6LChLEVEAIBks5crLE8wA%3D%3D)    |
| Bilateral     | Nothing(only normalize)    |
| ![](https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/biliteral.png?token=AVgtIkt8b5AT8TUHLKilJHFfXYciENd2ks5crLFfwA%3D%3D)    | ![](https://raw.githubusercontent.com/Nicetiesniceties/VFX/master/Hw1/Report/report_img/normalize.png?token=AVgtItYeUyxbuxA4aXuVPgAClghHnApeks5crLF7wA%3D%3D)    |
At the end, we choose Bilateral + Gamma as our final result due to our own aesthetic.


## Apendix

### Bilateal Filter
```python=
def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


def bilateral_filter(input_image, output_image, x, y, filter_size):
    w_sum = 0
    denoised_intensity = 0
    sigma_r = 100
    sigma_d = 200
    h = input_image.shape[0]
    w = input_image.shape[1]
    for i in range(-filter_size, filter_size + 1):
        for j in range(-filter_size, filter_size + 1):
            if x + i >= h or x + i < 0 or y + j >= w or y + j < 0:
                continue
            fr = gaussian(input_image[x + i][y + j] - input_image[x][y], sigma_r)
            gs = gaussian(np.linalg.norm([i, j]), sigma_d)
            w =  fr * gs
            denoised_intensity += w * input_image[x + i][y + j]
            w_sum += w
    denoised_intensity /= w_sum
    return denoised_intensity

def bilateral(input_image):
        filter_size = 2
        output_image = np.zeros(input_image.shape)
        h = input_image.shape[0]
        w = input_image.shape[1]
        for i in range(h):
            for j in range(w):
                output_image[i, j] = bilateral_filter(input_image, output_image,i, j, filter_size)
        return output_image

def bilateral_func(input_image):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    R = input_image[0]
    G = input_image[2]
    B = input_image[1]
    intensity = 0.2126 * R + 0.7152 * G + 0.0722 * B
    r = R / intensity
    g = G / intensity
    b = B / intensity
    log_intensity = np.log(intensity)
    log_base = bilateral(log_intensity)
    log_detail = log_intensity - log_base
    log_abs_scale = np.max(log_base) * compressionfactor
    log_output = log_base * compressionfactor + log_detail - log_abs_scale
    R_out = r * np.exp(log_output)
    G_out = g * np.exp(log_output)
    B_out = b * np.exp(log_output)
    plt.show()
    return [R_out, B_out, G_out]
```



