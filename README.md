# Discovering Google´s Magenta DDSP 

Digitally process audio data with ML &amp; Magenta 

Magenta: Open Source Research 

## Challenges   

### Representation of Audio
 
One song of 3 minutes : 1 Million time steps BUT relevant information is much less! **The art is to extract those featuers** and find a meaningful representation for music. If music is only structured as a bit stream consisting of 1´s and 0´s it is very difficult to know what´s going on.  

#### Phase Alignment

Phases of different simultaneous frequencies have to be aligned precisely else they cancel each other out or become to loud which leads to an overall bad mix.

#### Fourier based Models
Another widely used method was to just learn all the waveform packages, decompose them into sine and cosine waves and finally recreate the soundwave out of the Fourier waves. However, the waveforms overlap and therefore this procedure is inprecise and inefficient.   

#### Autogenerative Models
Autogenerative models try to mitigate these problems by constructing the waveform sample by sample so they do not suffer from the same bias the wave packets. 
Still, the waveform shapes do not perfectly correlate with human perception because of teacher enforcing / exposure bias during network training.
For example the waveforms on the right sound the same for humans but cause different perceptual losses for the model. Moreover they need alot of data to work. 

<img width="405" alt="ddsp_challenges_waveforms" src="https://user-images.githubusercontent.com/24375094/208299823-f1c3ce8c-39d0-4bb2-96dc-d0043be9c0e3.png"> 

#### Oscillation based Models

Instead of learning all the features for waveform/Short Fourier Transformation another more selective way would be to learn only the synthesis parameters. 
Still the model is prone to errors and lacks expressiveness when learning with these annotated synthesizer parameters. 

<img width="735" alt="annotated_synthesis_features" src="https://user-images.githubusercontent.com/24375094/208300159-41de5390-199c-4b90-bd7d-328f2d28b29a.png"> 

Rather than predicting the waveforms or Fourier coefficients those models directly generates the oscillations. These “analysis/synthesis” models use expert knowledge and hand-tuned heuristics to xtract synthesis parameters (analysis) that are interpretable (**loudness** in dB and **frequencies** in Hz) and can be
used by the generative algorithm (synthesis). With this representation you can represent a harmonic oscillation precisely solely by using the fundamental frequency (f0), some harmonics (integer multiplications) and the amplitude. 

<img width="481" alt="ddsp_harmonic_transformation" src="https://user-images.githubusercontent.com/24375094/208642273-5b044358-22cf-4526-92e7-1e517dc68d4b.png">


## Dataset 

### Sorting 

For our project we used the TensorfFlow GAN subset of the NSYNTH dataset. It offers preprocessed samples which contain the most relevant features (amplitude and frequency) ready to use with the DDSP library. 
For efficient training we *downloaded* the 60 gigabyte of 11 instrument samples instead of streaming them. Since the *data wasn´t storted by instrument type* we had to do this step additionally: We read the TFRecord files into python, parsed them to JSON to identify the instrument label and then wrote them back to TFRecord files. For this to work properly, we had to continuously remove the written objects from the memory such that it did not overflow.  
All in all this procedure took around 10 hours to sort the samples. 

#### Raw TFRecord String Representation 

<img width="788" alt="tfrecord_raw_string" src="https://user-images.githubusercontent.com/24375094/208647954-7a3f98de-d8fb-4b52-92b9-fac7517f3599.png">

#### TFRecord JSON Representation 


<img width="796" alt="tfrecord_json_representation" src="https://user-images.githubusercontent.com/24375094/208648732-bc3f69e8-90af-4db9-b16a-82f8f9488aa2.png">

### Feature Representation 
The features are presented as floatList tensors which contain the values over very small timesteps (e.g. length of 64000).
For efficient processing, (the features of) the input data has to be aligned with the architecture of a neural network.  

<img width="814" alt="feature_structure_gan" src="https://user-images.githubusercontent.com/24375094/208645745-041cb414-f287-45bb-8a47-8252fb813ad1.png">


## Training

That´s where the ddsp library comes in: it offers sound modules (synthesizers) which are differentiable and therefore can use back propagation to tune their synthesizer parameters (analog to recreating a sound on a synthesizer) and do not learn as much bias as the other models by the help of deep specialized and structured layers. Thanks to specialized layer types we have ***faster training of autoencoders*** and therefore quick feedback, which offers a more convenient workflow than waiting 16 hours for training to finish until you can implement further changes.

### Training of autoencoders
<img width="1638" alt="ddsp_autoencoder" src="https://user-images.githubusercontent.com/24375094/208653552-06a19ab8-fbaa-4c42-86fc-490c9ce4b0e8.png">

### Python Code
<img width="500", height="800", alt="colab_tut_training_basic_code_python_soundmodules" src="https://user-images.githubusercontent.com/24375094/208652789-f7b99ce7-d19c-435a-af41-4a02ec325554.png">





## Notes to Jesse DDSP video


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






  

