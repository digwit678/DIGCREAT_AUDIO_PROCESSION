<body name="ddsp">  
 
# Discovering Google´s Magenta DDSP 

Digitally process audio data with ML &amp; Magenta 


For the course Digital Creativity we explored the open source Google Magenta DDSP library. We decided to work mostly on Google Colab because it´s  much more convenient for us regarding installations, dependencies and training on GPU. 
The only exception to this is the dataset: We downloaded it to a local disk from Google Clouds and and also sorted it locally. There would be also notebooks on converting your own data to the needed format (TFR) when working with DDSP. Since we did not have enough of the right .wav data we used a TFR dataset with prepared MIDI samples. 

We accomodated ourselves to DDSP by going through a lot of the tutorials (<a href="https://github.com/magenta/ddsp/tree/main/ddsp/colab/tutorials">DDSP TUTORIALS</a> ). After that we used our gathered TFR data for small training on a single instrument type and then "predict" a sample of another instrument with the help of an adjusted DDSP notebook <a href="https://colab.research.google.com/github/magenta/ddsp/blob/main/ddsp/colab/tutorials/3_training.ipynb">this notebook</a> (see here).   "Prediction" in this sense means, you predict how that sample (e.g. a keyboard tone) would sound with the sound characteristics (timbre) of a different instrument (e.g. string) or simpler: How would a keyboard sound if it played strings ?  

# Citation 

All notebook sources in the folder ddsp_notebooks_adjusted belong to <a href="https://github.com/magenta/ddsp">Google Magenta´s DDSP</a> research team.  

@inproceedings{
  engel2020ddsp,
  title={DDSP: Differentiable Digital Signal Processing},
  author={Jesse Engel and Lamtharn (Hanoi) Hantrakul and Chenjie Gu and Adam Roberts},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=B1x1ma4tDr}
}



# Theory
## Challenges   

### Representation of Audio

<div name="representation">  
 
One song of 3 minutes : 1 Million time steps BUT relevant information is much less! **The art is to extract those featuers** and find a meaningful representation for music. If music is only structured as a bit stream consisting of 1´s and 0´s it is very difficult to know what´s going on.  </div>  

 
<div name = "bias">  
 
### Bias In Conventional Representations  

<img width="1000" height="350" alt="ddsp_challenges_waveforms" src="https://user-images.githubusercontent.com/24375094/208299823-f1c3ce8c-39d0-4bb2-96dc-d0043be9c0e3.png"> 

###  Phase Alignment  
<p>
For strided convolution waves are represented as overlapping frames, whereas in reality sound moves in different phases and would have to be aligned precisely between two fixed frames or else it would lead to bias.</p>

###  Fourier based Models  
<p>
Another widely used method was to just learn all the waveform packages, decompose them into sine and cosine waves and finally recreate the soundwave out of the Fourier waves. However, the waveforms overlap and therefore this procedure leads to bias again.  </p>

###  Autogenerative Models 
<p>
Autogenerative models try to mitigate these problems by constructing the waveform sample by sample so they do not suffer from the same bias the others do. </br>
However, the waveform shapes still do not perfectly correlate with human perception and get incoherently corrected during model training:</br>
For example the waveforms on the right sound the same for humans but cause different perceptual losses for the model. Moreover they need alot of data to work. </p>
</div>

<div name="oscillation">    
   
###  Back to the Roots: Oscillation based Models  

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
 <ul align="center"> Fundamental Frequency F0 (Hz) </ul>  
 <ul align="center"> Harmonics (F0 multiplications: odd, even, ...) </ul>  
 <ul align="center"> Amplitude (dB) </ul>
  <br></br> 
 </div>   
 This representation does not imply the model is completely free from bias but it seems to approach the nature and complexity of sound the best yet.  
</div>


<div name="data">  
 
## Dataset 
 
### Downloading
 
for more efficient training we downloaded our whole dataset from *Google Clouds* with the following link: https://console.cloud.google.com/storage/browser/tfds-data/datasets/nsynth;tab=objects?prefix=&forceOnObjectsSortingFiltering=false
 


to download multiple items at once you need to use gsutil (https://cloud.google.com/storage/docs/gsutil_install). This command requires to have parts of Google CLI installed on your computer  

 1.) install Google CLI (https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe)  
 2.) make sure gsutil is installed on Google CLI  
 3.) download files with gsutil command from terminal to storage location (recommended for big data amounts: external drive, e.g. "E:\gansynth") :  
 
   
     gsutil -m cp -r "gs://tfds-data/PATH" "STORAGE_PATH" 
  
![download_nsynth](https://user-images.githubusercontent.com/24375094/210249830-4ab42404-6a49-4c02-a450-cb11722f40c9.jpg)

### Sorting 
 
<p>
For our project we used the TensorfFlow GAN subset of the NSYNTH dataset. It offers preprocessed samples which contain the most relevant features (amplitude and frequency) ready to use with the DDSP library. </br>
For efficient training we <i> downloaded </i> the 60 gigabyte of 11 instrument samples instead of streaming them. Since the <i> data wasn´t storted by instrument type </i> we had to do this step additionally: We read the TFRecord files into python, parsed them to JSON to identify the instrument label and then wrote them back to TFRecord files. For this to work properly, we had to <b>continuously remove the written objects from the memory such that it did not overflow</b>.  
All in all this procedure took around 10 hours to sort the samples. </p>

<h4 align="center"> Raw TFRecord String Representation </h4> 

<p align="center"><img width="788" alt="tfrecord_raw_string" src="https://user-images.githubusercontent.com/24375094/208647954-7a3f98de-d8fb-4b52-92b9-fac7517f3599.png"></p>

<h4 align="center"> TFRecord JSON Representation </h4> 


<p align="center"><img width="796" alt="tfrecord_json_representation" src="https://user-images.githubusercontent.com/24375094/208648732-bc3f69e8-90af-4db9-b16a-82f8f9488aa2.png"></p>

<h4 align="center"> Adjusting Features names </h4> 

<p align="center">To get our TFR data working with the DDSP (e.g. notebook <i> training </i>) we had to adjust the classes slightly do accept the feature names with slashes instead of dashes (f0_hz = f0/hz) else we had to do the whole sorting process again to change feature names </p>

<h3 align="center">   Feature Representation </h3>
<p align="center">
The features are presented as floatList tensors which contain the values over very small timesteps (e.g. length of 64000). </br>  
For efficient processing, (the features of) the input data has to be aligned with the architecture of a neural network.  </p>

<p align="center"><img width="814" alt="feature_structure_gan" src="https://user-images.githubusercontent.com/24375094/208645745-041cb414-f287-45bb-8a47-8252fb813ad1.png"></p>
</div>


<div name="training">  
 
## Training
<p>
DDSP achitecture is based on a transformer network.  </br>
That´s where the ddsp library comes in: it offers sound modules (synthesizers) which are differentiable and therefore can use back propagation to tune their synthesizer parameters (analog to recreating a sound on a synthesizer) and do not learn as much bias as the other models by the help of deep specialized and structured layers. </br>
Thanks to these layer types we have <b><i>faster training of autoencoders</i></b> and therefore quick feedback, which offers a <i>more instrument like workflow</i> than iterating for 16 hours of training until you can implement further changes.</p>

<h3 align="center">  Training of Autoencoders </h3> 
<img width="1638" alt="ddsp_autoencoder" src="https://user-images.githubusercontent.com/24375094/208653552-06a19ab8-fbaa-4c42-86fc-490c9ce4b0e8.png">

<h3 align="center">Python Code <h3 align="center">
<p align="center"><img width="500" height="800" alt="colab_tut_training_basic_code_python_soundmodules" src="https://user-images.githubusercontent.com/24375094/208652789-f7b99ce7-d19c-435a-af41-4a02ec325554.png"></p>


<p align="center"> After training of 300 trainingsteps ( ~ 30 seconds) we receive following synth representation of the input. </p>
<p align="center"><img width="530" alt="ddsp_input_features_synth_parameters" src="https://user-images.githubusercontent.com/24375094/208658561-4a6da72b-4598-44ca-add5-08266a4f71de.png"></p>
 </div>
 </body>



## Notes to Jesse DDSP video (to be removed) 


Lowest level: 
More semantically meaningful: symbolic modelling as a language model --> faster workflow (fast feedback) in comparison to training your model for 16 hours and then see what happened.


old: predict new waveform, given old waveform (we perceive different waveforms as the same sound), model the waveforms sample by sample 
idea: incorporate prior information of ddsp and signal processing ==> generate simple oscillator components: whats frequencies and amplitudes of a model (interprete signal processes in TensorFlow , take gradient and look at frequency alignment of input and target audio to create expressive sound): manipulate frequencies individually in selectively with knowing whats going on in the inside of the model (interpretable),  (ear (ear is sensitive to phases of sinewaves if there is no frequency): 1.) frequency decomposition)  
creativity: What if I take the pitch and loudness of a different signal than the one my decoder is trained on ? 
Pitch and frequency stays the same but the tone changes according to the instrument learned by the decoder (decodes pitch and freq into e.g. a violin sound) (timbre transfer). DDSP allows to run models in real time (was never possible with raw waveform models)  

You can get very creative by trying routing lots of differents submodules in different ways. 

Training of autoencoders does not need a lot of data and usually only takes around 10 minutes: supervised by itself (todo?) 

Symbolic Representation 

Prior knowledge: how to decompose the grammar of music which works for a certain context. The philosophy is to incorporate this language and create new contexts
As a musician this gives you the opportunity to incorporate your own sounds and turn them into something new, but also for "non musician people" who like the thrill of experimenting and creating. 






  

