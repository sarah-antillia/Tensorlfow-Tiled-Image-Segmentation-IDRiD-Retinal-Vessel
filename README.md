<h2>Tensorflow-Tiled-Image-Segmentation-IDRiD-Retinal-Vessel (2025/03/09)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <a href="/"><b>IDRiD Retinal Vessel</b></a>
 based on Pretrained HRF Retinal Vessel Model, which was trained by 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a <b>pre-augmented tiled dataset</b> <a href="https://drive.google.com/file/d/1bCbZRej3_aOaYuvXbv0vYnrPold3aXPf/view?usp=sharing">
Augmented-Tiled-HRF-ImageMask-Dataset.zip</a>, which was derived by us from the following dataset:<br><br>
<a href="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip">
Download the whole dataset (~73 Mb)</a> in <a href="https://www5.cs.fau.de/research/data/fundus-images/"><b>High-Resolution Fundus (HRF) Image Database</b></a>.
<br>
<br>
Please see also our experiments:<br>
<li>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel</a> based on 
<a href="https://www5.cs.fau.de/research/data/fundus-images/">High-Resolution Fundus (HRF) Image Database</a>.
</li>
<li>
<a href="https://github.com/sarah-antillia/Tensorlfow-Tiled-Image-Segmentation-Pre-Augmented-DRIVE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-DRIVE-Retinal-Vessel</a> based on 
<a href="https://drive.grand-challenge.org/">DRIVE: Digital Retinal Images for Vessel Extraction</a>
</li>
<li>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</a> based on 
<a href="https://cecas.clemson.edu/~ahoover/stare/">STructured Analysis of the Retina</a>.
<br>
</li>
<li>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel">
Tensorflow-Image-Segmentation-Retinal-Vessel</a> based on <a href="https://researchdata.kingston.ac.uk/96/">CHASE_DB1 dataset</a>.
</li>
<br>
<br>
<b>Experiment Strategies</b><br>
<br>
<b>1. Masks (antillia ground truth) for IDRiD master Original Images</b><br>
The IDRiD dataset contains no masks (ground truths) data for Retinal Vessel. Therefore, we created our own master masks (ground truths) from the 
IDRiD Original Images by using Tiled Image Segmentation method and Pretrained-HRF-Retinal-Vessel 
UNet Model which was trained by 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a <b>pre-augmented tiled dataset</b> <a href="https://drive.google.com/file/d/1bCbZRej3_aOaYuvXbv0vYnrPold3aXPf/view?usp=sharing">
Augmented-Tiled-HRF-ImageMask-Dataset.zip</a>.
<br><br>
<b>2. Augment IDRiD Dataset </b><br>
We augmented the IDRiD-master images and masks by using ImageMaskDatasetGenerator. 
<br><br>

<b>3. Split Augmented-IDRiD-master </b><br>
We splitted the Augmented-IDRiD-master images and masks into test, train and valid subsets. 
<br><br>
<b>4. Train IDRiD Retinal Vessel Segmentation Model </b><br>
We trained and validated a TensorFlow UNet model by using the <b>Augmented-IDRiD train and valid subsets</b>
<br>
<br>
<b>5. Evaluate IDRiD Segmentation Model </b><br>
We evaluated the performance of the trained UNet model by using the <b>Augmented-IDRiD test</b> dataset
 by computing the <b>bce_dice_loss</b> and <b>dice_coef</b>. </b>
<br>
<br>
<b>5. Tiled Inference   </b><br>
We applied our Tiled Image Segmentation method to infer the Retinal Vessel for the mini_test images 
of the IDRiD images of 4288x2848 pixels.<br><br>

<hr>
<b>Actual Tiled Image Segmentation for IDRiD images of 4288x2848 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (antillia ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/images/10005.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/masks/10005.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test_output_tiled/10005.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/images/10053.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/masks/10053.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test_output_tiled/10053.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/images/barrdistorted_1001_0.3_0.3_10004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/masks/barrdistorted_1001_0.3_0.3_10004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test_output_tiled/barrdistorted_1001_0.3_0.3_10004.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this HRFSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
<h3>1.1 <b>High-Resolution Fundus (HRF) Image Database</b></h3>

The dataset used here has been taken from the dataset 
<a href="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip">
Download the whole dataset (~73 Mb)</a> in <a href="https://www5.cs.fau.de/research/data/fundus-images/"><b>High-Resolution Fundus (HRF) Image Databaset</b></a>.
<br><br>
<b>Introduction</b><br>
This database has been established by a collaborative research group to support comparative studies on 
automatic segmentation algorithms on retinal fundus images. The database will be 
iteratively extended and the webpage will be improved.<br>
We would like to help researchers in the evaluation of segmentation algorithms. 
We encourage anyone working with segmentation algorithms who found our database useful to send us 
their evaluation results with a reference to a paper where it is described. This way we can extend our database of algorithms with the given results to keep it always up-to-date.
<br>
The database can be used freely for research purposes. We release it under Creative Commons 4.0 Attribution License. 
<br><br>
<b>Citation</b><br>
<a href="https://onlinelibrary.wiley.com/doi/10.1155/2013/154860">
<b>Robust Vessel Segmentation in Fundus Images</b></a>
<br>
Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.<br>
<b>
International Journal of Biomedical Imaging, vol. 2013, 2013</b>
<br>
<h3>
<h3>1.2 <b>IDRiD Dataset</b></h3>
The dataset used here has been take from the following <b>IEEE DataPort</b> web site<br>

<a href="https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid">
<b>
Indian Diabetic Retinopathy Image Dataset (IDRiD)
</b>
</a>
<br><br>
Please see also <a href="https://idrid.grand-challenge.org/">
<b>DIABETIC RETINOPATHY: SEGMENNTATION AND GRAND CHALLENGE</b> </a>

<br>
<br>
<b>Citation Author(s):</b><br>
Prasanna Porwal, Samiksha Pachade, Ravi Kamble, Manesh Kokare, Girish Deshmukh, <br>
Vivek Sahasrabuddhe, Fabrice Meriaudeau,<br>
April 24, 2018, "Indian Diabetic Retinopathy Image Dataset (IDRiD)", IEEE Dataport, <br>
<br>
DOI: <a href="https://dx.doi.org/10.21227/H25W98">https://dx.doi.org/10.21227/H25W98</a><br>
<br>

<b>License:</b><br>
<a href="http://creativecommons.org/licenses/by/4.0/">
Creative Commons Attribution 4.0 International License.
</a>
<br>


<h3>2. Create Masks for IDRiD Retinal Vessel Images</h3>
<h3>2.1 Download IDRiD IMAGES Dataset</h3>
Please download 
<a href="https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid">
<b>
Indian Diabetic Retinopathy Image Dataset (IDRiD)
</b>
</a>
The folder structure of the original Segmentation data is the following.<br>
<pre>
./A. Segmentation
├─1. Original Images
│  ├─a. Training Set
│  └─b. Testing Set
└─2. All Segmentation Groundtruths
   ├─a. Training Set
   │  ├─1. Microaneurysms
   │  ├─2. Haemorrhages
   │  ├─3. Hard Exudates
   │  ├─4. Soft Exudates
   │  └─5. Optic Disc
   └─b. Testing Set
       ├─1. Microaneurysms
       ├─2. Haemorrhages
       ├─3. Hard Exudates
       ├─4. Soft Exudates
       └─5. Optic Disc
</pre>
As shown above, this datatset contains no masks (ground truths) data for <b>IDRiD Retinal Vessel</b>.
Therefore, we created our own mask files for the IDRiD Retinal Vessel.<br>
At first, please copy all image files in <b>1. Original Images</b> under the following folder <b>images</b>.<br>
<pre>
./projects
 └─generator
    └─IDRiD-master
        └─images
</pre>

<h3>2.2 Download Augmented-Tiled-HRF-Pretrained-Model</h3>

Please download our <a href="https://drive.google.com/file/d/1xsfPzQ8srbKr8qXo-mDyUJrne4FRG5nw/view?usp=sharing">
Augmented-Tiled-HRF-Pretrained-Model.zip,</a> expand it and place <b>best_model.h5</b> under models folder as shown below. 
<pre>
./projects
 └─TensorflowSlightlyFlexibleUNet
    └─Augmented-Tiled-HRF
         └─models
             └─best_model.h5
</pre>
<h3>2.3 Run Tiled Inference method</h3>

Please move the folder "./projects/TensorflowSlightlyFlexibleUNet/Augmented-Tiled-HRF". and run the following bat file.
<pre>
5.tiled_infer-idrid.bat
</pre>
This will generate our own masks (ground truths) for Retinal Vessel of IDRiD images by using the HRF-Pretrained-Model,
without any human experts.<br>
<pre>
./projects
 └─generator
    └─IDRiD-master
        ├─images
        └─masks
</pre>
The number of images and their corresponding masks in IDRiD-master is 81 respectively, which is too small to use for a training set
of our Segmentation Model.<br>
<br>

<h3>2.4 Augment IDRiD-master</h3>
Please move to "./projects/generator" folder and run the following Python script.<br>
<pre>
python ImageMaskDatasetGenerator.py
</pre>
, by which the following Augmented-IDRiD-master will be generated.
<pre>
./projects
 └─generator
    └─Augmented-IDRiD-master
        ├─images
        └─mask
</pre>


<h3>2.5 Split Augmented-IDRiD-master</h3>
Please move to "./projects/generator" folder and run the following Python script.<br>
<pre>
python spit_augmented_master.py
</pre>
, by which the following Augmented-IDRiD dataset will be created.
<pre>
./dataset
└─Augmented-IDRiD
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
This is a Non-Tiled 4288x2848 pixels images and their corresponding masks dataset for IDRiD Retinal Vessel.<br>

You may also download this dataset from the google drive <a href="https://drive.google.com/file/d/16bY4Kr4lLWAbK8t9SK12JG8gizdEba8r/view?usp=sharing">Antillia-Augmented-IDRiD-Dataset.zip (2.06G)</a>
<br>
<b>Augmented-IDRiD Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/Augmented-IDRiD_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large, but enough 
to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained HRF TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/IDRiD and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Enabled Batch Normalization.<br>
Defined a small <b>base_filters=16</b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Dataset class</b><br>
Specified ImageMaskDataset class.
<pre>
[dataset]
datasetclass  = "ImageMaskDataset"
resize_interpolation = "cv2.INTER_LINEAR"
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Tiled Inference</b><br>
Used the original IDRiD IMAGES as a mini_test dataset for our inference images.
<pre>
[tiledinfer] 
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output_tiled"
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer      = False
epoch_change_infer_dir  = "./epoch_change_infer"
epoch_change_tiledinfer = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images       = 6
</pre>

By using this callback, on every epoch_change, the epoch change tiled inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_tiled_inference output at starting (1,2,3,4)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/asset/epoch_change_tiled_infer_at_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_tiled_inference output at ending (75,76,77,78)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/asset/epoch_change_tiled_infer_at_end.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was terminated at epoch 100.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/asset/train_console_output_at_epoch_100.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/IDRiD</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for IDRiD/test.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/asset/evaluate_console_output_at_epoch_100.png" width="720" height="auto">
<br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this IDRiD/test was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.0845
dice_coef,0.8859
</pre>
<br>

<h3>
5 Tiled Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/IDRiD</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for IDRiD.<br>
<pre>
./4.tiledinfer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images (4288x2848 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(antillia ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled inferred test masks (4288x2848 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 4288x2848 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (antillia ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/images/10005.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/masks/10005.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test_output_tiled/10005.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/images/10020.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/masks/10020.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test_output_tiled/10020.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/images/10053.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/masks/10053.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test_output_tiled/10053.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/images/barrdistorted_1001_0.3_0.3_10004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/masks/barrdistorted_1001_0.3_0.3_10004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test_output_tiled/barrdistorted_1001_0.3_0.3_10004.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/images/barrdistorted_1002_0.3_0.3_10041.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/masks/barrdistorted_1002_0.3_0.3_10041.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test_output_tiled/barrdistorted_1002_0.3_0.3_10041.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/images/barrdistorted_1002_0.3_0.3_10080.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test/masks/barrdistorted_1002_0.3_0.3_10080.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Augmented-IDRiD/mini_test_output_tiled/barrdistorted_1002_0.3_0.3_10080.jpg" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. Locating Blood Vessels in Retinal Images by Piecewise Threshold Probing of a Matched Filter Response</b><br>
Adam Hoover, Valentina Kouznetsova, and Michael Goldbaum<br>
<a href="https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf">
https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf
</a>
<br>
<br>
<b>2. High-Resolution Fundus (HRF) Image Database</b><br>
Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.<br>
<a href="https://www5.cs.fau.de/research/data/fundus-images/">
https://www5.cs.fau.de/research/data/fundus-images/
</a>.
<br>
<br>
<b>3. Robust Vessel Segmentation in Fundus Images</b><br>
Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.<br>

<a href="https://onlinelibrary.wiley.com/doi/10.1155/2013/154860">
https://onlinelibrary.wiley.com/doi/10.1155/2013/154860
</a>
<br>
<br>
<b>4. State-of-the-art retinal vessel segmentation with minimalistic models</b><br>
Adrian Galdran, André Anjos, José Dolz, Hadi Chakor, Hervé Lombaert & Ismail Ben Ayed<br>
<a href="https://www.nature.com/articles/s41598-022-09675-y">
https://www.nature.com/articles/s41598-022-09675-y
</a>
<br>
<br>
<b>5. Retinal blood vessel segmentation using a deep learning method based on modified U-NET model</b><br>
Sanjeewani, Arun Kumar Yadav, Mohd Akbar, Mohit Kumar, Divakar Yadav<br>
<a href="https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3">
https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3</a>
<br>
<br>
<b>6. Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</a>
<br>
<br>
<b>7. Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel</a>
<br>
<br>
