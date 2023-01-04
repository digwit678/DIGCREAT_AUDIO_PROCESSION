<body name="ddsp">  
 
# Discovering Google Magenta´s DDSP library

***Digitally process audio data with ML*** 

 
## Introduction
For the course *Digital Creativity* we explored the open source library **Google Magenta DDSP**.  
We decided to work mostly on Google Colab because it´s  much more convenient for us regarding installations, dependencies and training on GPU. The only exception to this is working with the dataset: It was all downloaded from [Google Clouds](https://console.cloud.google.com/storage/browser/tfds-data/datasets/nsynth/gansynth_subset.f0_and_loudness/2.3.3?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false) to a local disk  and and sorted there [by using this notebook](https://github.com/digwit678/DIGCREAT_AUDIO_PROCESSION/blob/main/data/data_sorting/NSYNTH-TFRECORD-SORT.ipynb).  
There  are arlready notebooks on converting your own wave data to the needed format (TFR) when working with DDSP. Since we did not have enough of the right wave data we used a TFR dataset with prepared MIDI samples. 

We accomodated ourselves to DDSP by going through a lot of the tutorials (<a href="https://github.com/magenta/ddsp/tree/main/ddsp/colab/tutorials">DDSP TUTORIALS</a> ).     
 Afterwards we used our gathered and sorted TFR data for small training on a single instrument type and then *predict* a sample of another instrument with the help of an [adjusted DDSP NOTEBOOK (we recommend working with this version for reproduction, continuation etc.)](https://github.com/digwit678/DIGCREAT_AUDIO_PROCESSION/blob/main/ddsp_notebooks_adjusted/small_training_trials/3_training_string_340_keyboard_harmonic.ipynb) .   *Prediction* in this sense means, you predict how that sample (e.g. a keyboard tone) would sound with the sound characteristics (timbre) of a different instrument (e.g. string) or simpler: ***How would a keyboard tone sound if it played with a string sound/timbre ?***  





## Theory

 ### <i>Challenge: Representation of Audio</i>

<div name="representation">  
 
One song of 3 minutes : 1 Million time steps **BUT** relevant information is much less! ***The art is to extract those featuers*** and find a meaningful representation for music. If music is only structured as a bit stream consisting of 1´s and 0´s it is very difficult to know what´s going on.  </div>  

 
<div name = "bias">  
 
### <i>Bias In Conventional Representations</i>  

<img width="1000" height="350" alt="ddsp_challenges_waveforms" src="https://user-images.githubusercontent.com/24375094/208299823-f1c3ce8c-39d0-4bb2-96dc-d0043be9c0e3.png"> 

###  <i>Phase Alignment</i>  
<p>
For strided convolution waves are represented as overlapping frames, whereas in reality sound moves in different phases and would have to be aligned precisely between two fixed frames or else it would lead to bias.</p>

###  <i>Fourier based Models</i>  
<p>
Another widely used method was to just learn all the waveform packages, decompose them into sine and cosine waves and finally recreate the soundwave out of the Fourier waves. However, the waveforms overlap and therefore this procedure leads to bias again.  </p>

###  <i>Autogenerative Models</i> 
<p>
Autogenerative models try to mitigate these problems by constructing the waveform sample by sample so they do not suffer from the same bias the others do. </br>
However, the waveform shapes still do not perfectly correlate with human perception and get incoherently corrected during model training:</br>
For example the waveforms on the right sound the same for humans but cause different perceptual losses for the model. Moreover they need alot of data to work. </p>
</div>

<div name="oscillation">    
   
###  <i>Back to the Roots: Oscillation based Models</i>  

 ![oscillations](https://user-images.githubusercontent.com/24375094/209557212-ead2037b-8d1d-4eaf-8e4d-ccb0d0fa6801.png)  
 
<p>   
Oscillation is defined as <i> the process of repeating variations of any quantity or measure about its equilibrium value in time </i>.  </br>
Most of the things in nature oscillate (vibrate) at a characteristic (natural) frequency or frequencies.   </br>
Some familiar examples are the motions of the pendulum of a clock and playground swing, up and down motion of small boats, ocean waves, and motion of the string or reeds on musical instruments.</p>  </br>

<img width="1000" alt="annotated_synthesis_features" src="https://user-images.githubusercontent.com/24375094/208300159-41de5390-199c-4b90-bd7d-328f2d28b29a.png">   
<p>  

Rather than predicting the waveforms or Fourier coefficients those models directly generates the oscillations.    
These <i>analysis/synthesis</i> models use expert knowledge and hand-tuned heuristics to xtract synthesis parameters (<i>analysis</i>) that are interpretable (<b> loudness</b> in dB and <b>frequencies</b> in Hz) and can be used by the generative algorithm (<i>synthesis</i>).</p>      

<img width="1000"  height="400" alt="ddsp_harmonic_transformation" src="https://user-images.githubusercontent.com/24375094/208642273-5b044358-22cf-4526-92e7-1e517dc68d4b.png">  
<br></br>

<div align="center">
With this features you <i> can represent a harmonic oscillation precisely solely by using </i>
 
 <br></br>
 <ul align="center"> <b>Fundamental Frequency F0 </b> (Hz) </ul>  
 <ul align="center"> <b>Harmonics</b> (F0 multiplications: odd, even, ...) </ul>  
 <ul align="center"> <b>Amplitude</b> (dB) </ul>
  <br></br> 
 </div>   
 This representation does not imply the model is completely free from bias but it seems to approach the nature and complexity of sound the best yet.  
</div>


<div name="data">  
 
## Dataset 


For our first trial we used the [nsynth/full dataset](https://console.cloud.google.com/storage/browser/tfds-data/datasets/nsynth/full?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false) but then realized the features weren´t optimally suited for working with DDSP so we changed to the [nsynth/gansynth_subset.f0_and_loudness/2.3.3.](https://console.cloud.google.com/storage/browser/tfds-data/datasets/nsynth/gansynth_subset.f0_and_loudness/2.3.3) with f0 and loudness features which were missing.   

 
### <i>Download</i>

If you would like to try out a training on a single instrument without downloading the whole dataset we uploaded two TFR files (containing a lot of samples!) for string and keyboard ([data folder](https://github.com/digwit678/DIGCREAT_AUDIO_PROCESSION/tree/main/data)) and used up our whole LF/Large File Storage Github resources. The data there should be enough to train on strings samples (xor keyboard) and predict with one keyboard sample (xor string)

for more efficient training we downloaded the whole GANSYNTH 2.3.3. (sub) dataset from *Google Clouds* with [this link](https://console.cloud.google.com/storage/browser/tfds-data/datasets/nsynth;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)
 


to download multiple items at once you need to [use gsutil](https://cloud.google.com/storage/docs/gsutil_install). This command requires to have parts of Google CLI installed on your computer  

 1.) [install Google CLI](https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe)    
 2.) make sure gsutil is installed on Google CLI (e.g. try ```gsutil ls``` in command prompt: is the command recognized?)  
 3.) download files with gsutil command from the cloud to a (local) storage location (external drive, e.g. "E:\gansynth", recommended for big data amounts:    
 
   
     gsutil -m cp -r "gs://tfds-data/PATH" "STORAGE_PATH" 
  
![download_nsynth](https://user-images.githubusercontent.com/24375094/210249830-4ab42404-6a49-4c02-a450-cb11722f40c9.jpg)

### <i>Sorting</i> 
 
<p>
For our project we used the TensorfFlow GAN subset of the NSYNTH dataset. It offers preprocessed samples which contain the most relevant features (amplitude and frequency) ready to use with the DDSP library. </br>
For efficient training we <i> downloaded </i> the 11 instrument samples instead of streaming them. Since the <i> data wasn´t storted by instrument type </i> we had to do this step additionally to observe the effects of training on a single instrument type.   
We read the TFRecord files into Python, parsed them to JSON to identify the instrument label and then wrote them back to TFRecord files with <a href="https://github.com/digwit678/DIGCREAT_AUDIO_PROCESSION/blob/main/data/data_sorting/NSYNTH-TFRECORD-SORT.ipynb">the help of this notebook</a>. For this to work properly, we had to <b>continuously remove the written objects from the memory such that it did not overflow</b>.  
All in all this procedure took around 10 hours to sort the samples for the first dataset and then significantly less time for the second (samller) set (30-60 minutes). </p>

<h4 align="center"> <i>Raw TFRecord String Representation</i> </h4> 

<p align="center"><img width="788" alt="tfrecord_raw_string" src="https://user-images.githubusercontent.com/24375094/208647954-7a3f98de-d8fb-4b52-92b9-fac7517f3599.png"></p>

<h4 align="center"> <i>TFRecord JSON Representation</i> </h4> 


<p align="center"><img width="796" alt="tfrecord_json_representation" src="https://user-images.githubusercontent.com/24375094/208648732-bc3f69e8-90af-4db9-b16a-82f8f9488aa2.png"></p>

<h4 align="center"> <i>Adjusting Features names</i> </h4> 

<p align="center">To get our TFR data working with the DDSP (e.g. notebook <i> training </i>) we had to adjust the classes slightly do accept the feature names with slashes instead of dashes (f0_hz = f0/hz) else we had to do the whole sorting process again to change feature names </p>

<h4 align="center">   <i>Feature Representation</i> </h4>
<p align="center">
The features are presented as floatList tensors which contain the values over very small timesteps (e.g. length of 64000). </br>  
For efficient processing, (the features of) the input data has to be aligned with the architecture of a neural network.  </p>

<p align="center"><img width="814" alt="feature_structure_gan" src="https://user-images.githubusercontent.com/24375094/208645745-041cb414-f287-45bb-8a47-8252fb813ad1.png"></p>
</div>


<div name="training">  
 
## Training
<p>
DDSP achitecture is based on a transformer network.  </br>
That´s where the DDSP library comes in: it offers sound modules (synthesizers) which are differentiable and therefore can use back propagation to tune their synthesizer parameters (analog to recreating a sound on a synthesizer) and do not learn as much bias as the other models by the help of deep specialized and structured layers. </br>
Thanks to these layer types we have <b><i>faster training of autoencoders</i></b> and therefore quick feedback, which offers a <i>more instrument like workflow</i> than iterating for 16 hours of training until you can implement further changes.</p>

<h3 align="center"> <i>Training of Autoencoders</i> </h3> 
<p align="center"><img width="1638" alt="ddsp_autoencoder" src="https://user-images.githubusercontent.com/24375094/208653552-06a19ab8-fbaa-4c42-86fc-490c9ce4b0e8.png"></p>
<p align="center"> 
 For training on a TFR dataset we recommend using 
<a href="https://github.com/digwit678/DIGCREAT_AUDIO_PROCESSION/blob/main/ddsp_notebooks_adjusted/small_training_trials/3_training_string_340_keyboard_harmonic.ipynb"> this notebook</a> as a starting point 
</p>


<h3 align="center"> <i>Python Code for Layers/Synths</i> </h3>
<p align="center"><img width="500" height="800" alt="colab_tut_training_basic_code_python_soundmodules" src="https://user-images.githubusercontent.com/24375094/208652789-f7b99ce7-d19c-435a-af41-4a02ec325554.png"></p>

<h3 align="center"> <i>Results</i> </h3>
<p> We received the following outputs when training with 3 different synthesizers (= neural layers) trained on the same string data (until learning curve flattening, usually around 4.5-5) and predicted on the same keyboard sample </p>


<p align="center"><img width="522" alt="harmonic_training_string" src="https://user-images.githubusercontent.com/24375094/210359126-43c3820a-a45d-4dc1-8f8c-b4a73ef08485.png"><p align="center"><a href="https://github.com/digwit678/DIGCREAT_AUDIO_PROCESSION/blob/main/ddsp_notebooks_adjusted/small_training_trials/3_training_string_340_keyboard_harmonic.ipynb">Harmonic Synthesizer</a></p></p>
<p align="center"><img width="538" alt="sinusoid_training_string" src="https://user-images.githubusercontent.com/24375094/210359127-983f015c-11cf-40d9-a92c-ed38f05dcc03.png"><p align="center"><a href="https://github.com/digwit678/DIGCREAT_AUDIO_PROCESSION/blob/main/ddsp_notebooks_adjusted/small_training_trials/3_training_string_900_malletpd_sinusoid.ipynb">Sinusoid Synthesizer</a></p></p>
<p align="center"><img width="517" alt="wavetable_training_string" src="https://user-images.githubusercontent.com/24375094/210359129-a2872998-8b38-4b5a-a865-983b7a5e13df.png"><p align="center"><a href="https://github.com/digwit678/DIGCREAT_AUDIO_PROCESSION/blob/main/ddsp_notebooks_adjusted/small_training_trials/3_training_string_300_wt.ipynb">Wavetable Synthesizer</a></p></p>
<br></br>
<br></br>

<p> <i> We can observe from the spectograms that the harmonic synthesizer - as you´d probably expected - has the richest harmonic distribution </i> </p>
<br></br>
 </div>
 
## Possible Next Steps

Since the time for this project was scarce and the complexity relatively high we did not yet complete a full big training. To continue with the gathered data and lessons learned from a small training on a singular instrument, options for long training would be: 

- try bigger training on the timbre transfer notebook
- train a VST on the VST notebook 
- ...

we also prepared [the timbre transfer notebook](https://github.com/digwit678/DIGCREAT_AUDIO_PROCESSION/tree/main/ddsp_notebooks_adjusted/notebooks_to_continue) since the original version with the updated dependencies did not work. 

for more content, just have a look at [ddsp demos](https://github.com/magenta/ddsp/tree/main/ddsp/colab/demos): there are a lot of (new) ideas once your familiar with the library and the data!
<br></br>


## Further Links

[Youtube: Google staff research scientist Jesse Engel explaining DDSP](https://www.youtube.com/watch?v=20vUaDblkUM&ab_channel=TheTWIMLAIPodcastwithSamCharrington)
<br></br>

## Citation 

All notebook sources in the folder ddsp_notebooks_adjusted belong to <a href="https://github.com/magenta/ddsp">Google Magenta´s DDSP</a> research team.

<br></br>

```
# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
```

 
```
@inproceedings{  
  engel2020ddsp,  
  title={DDSP: Differentiable Digital Signal Processing},  
  author={Jesse Engel and Lamtharn (Hanoi) Hantrakul and Chenjie Gu and Adam Roberts},  
  booktitle={International Conference on Learning Representations},  
  year={2020},  
  url={https://openreview.net/forum?id=B1x1ma4tDr}  
}  
```
<br></br>

### <i>Notebook Sources</i>  

 <a href="https://colab.research.google.com/github/magenta/ddsp/blob/main/ddsp/colab/tutorials/3_training.ipynb">training on single instrument notebook</a>  
 <a href="https://github.com/magenta/ddsp/blob/main/ddsp/colab/demos/timbre_transfer.ipynb">timbre transfer notebook</a>
<br></br>

## Picture Sources (README & presentation) 

[DDSP paper](https://openreview.net/pdf?id=B1x1ma4tDr)

[playground oscillation](https://www.britannica.com/technology/pendulum)

[waves oscillation](https://www.google.com/search?q=waves&tbm=isch&ved=2ahUKEwiW98n92Kv8AhUzw7sIHaxHCpQQ2-cCegQIABAA&oq=waves&gs_lcp=CgNpbWcQAzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIFCAAQgAQyBQgAEIAEOggIABCABBCxA1DPBFiBCGDCCmgAcAB4AIABSIgBhQOSAQE2mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=9Ea0Y5atJ7OG7_UPrI-poAk&bih=881&biw=1824&rlz=1C1JJTC_enCH974CH975)


</body>






  

