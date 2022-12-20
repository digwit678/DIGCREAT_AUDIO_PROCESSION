# Discovering Google´s Magenta DDSP 

Digitally process audio data with ML &amp; Magenta 

Magenta: Open Source Research 

## Challenges    
 
One song of 3 minutes : 1 Million time steps BUT relevant information is much less! **The art is to extract those featuers** and find a meaningful representation for music. If music is only structured as a bit stream consisting of 1´s and 0´s it is very difficult to know what´s going on.  

### Phase Alignment

Phases of different simultaneous frequencies have to be aligned precisely else they cancel each other out or become to loud which leads to an overall bad mix.

### Fourier based Models
Another widely used method was to just learn all the waveform packages, decompose them into sine and cosine waves and finally recreate the soundwave by the means of waveform addition. However, the waveforms overlap and therefore this procedure is inprecise and inefficient.   

### Autogenerative Models
Autogenerative models try to mitigate these problems by constructing the waveform sample by sample so they do not suffer from the same bias the wave packets. 
Still, the waveform shapes do not perfectly correlate with human perception because of teacher enforcing / exposure bias during network training.
For example the waveforms on the right sound the same for humans but cause different perceptual losses for the model. Moreover they need alot of data to work. 

<img width="405" alt="ddsp_challenges_waveforms" src="https://user-images.githubusercontent.com/24375094/208299823-f1c3ce8c-39d0-4bb2-96dc-d0043be9c0e3.png"> 

### Oscillation based Models

Rather than predicting the waveforms or Fourier coefficients those models directly generates the oscillations. These “analysis/synthesis” models use expert knowledge and hand-tuned heuristics to xtract synthesis parameters (analysis) that are interpretable (**loudness** in dB and **frequencies** in Hz) and can be
used by the generative algorithm (synthesis). That´s where the ddsp library comes in: it offers sound modules (synthesizers) which are differentiable and therefore can use back propagation to tune their synthesizer parameters (analog to recreating a sound on a synthesizer). 


The features are presented as tensors 
For efficient processing, (the features of) the input data has to be aligned with the architecture of a neural network.  


Instead of learning all the features for waveform/Short Fourier Transformation another more selective way would be to learn only the synthesis parameters. 
Still the model is prone to errors and lacks expressiveness when learning with these annotated synthesizer parameters. 

<img width="735" alt="annotated_synthesis_features" src="https://user-images.githubusercontent.com/24375094/208300159-41de5390-199c-4b90-bd7d-328f2d28b29a.png"> 

Python library DDSP provides differentiable sound modules (wavetable synthesizer, harmonic synthesizer, reverb) whose parameters can be learned and improved by backwards propagation from the target waveform.
NNs do not have to learn the intrinsic biases with the help of deep specialized layers. 

Main features: main frequency, loudness

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






  

