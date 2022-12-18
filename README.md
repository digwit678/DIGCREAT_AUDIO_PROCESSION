# DIGCREAT_AUDIO_PROCESSION 
Digitally process audio data with ML &amp; Magenta 

Magenta: Open Source Research 

Challenge   

One song of 3 minutes : 1 Million time steps BUT relevant information is much less! The art is to extract those featuers 

Input data of neural network for audio synthesis has to be aligned with its architecture to allow for efficient procession. 


<img width="405" alt="ddsp_challenges_waveforms" src="https://user-images.githubusercontent.com/24375094/208299823-f1c3ce8c-39d0-4bb2-96dc-d0043be9c0e3.png">  

Instead of learning all the features for waveform/Short Fourier Transformation another more selective way would be to learn only the synthesis parameters. 
Still the model is prone to errors and lacks expressiveness when learning with these annotated synthesizer parameters. 

<img width="735" alt="annotated_synthesis_features" src="https://user-images.githubusercontent.com/24375094/208300159-41de5390-199c-4b90-bd7d-328f2d28b29a.png"> 

Python library DDSP provides differentiable sound modules (wavetable synthesizer, harmonic synthesizer, reverb) whose parameters can be learned and improved by backwards propagation from the target waveform.
NNs do not have to learn the intrinsic biases with the help of deep specialized layers. 

Main features: main frequency, loudness

Lowest level: 
More semantically meaningful: symbolic modelling as a language model --> faster workflow (fast feedback) in comparison to training your model for 16 hours and then see what happened.

idea: represent music in more meaningful way 
old: predict new waveform, given old waveform (we perceive different waveforms as the same sound), model the waveforms sample by sample 
idea: incorporate prior information of ddsp and signal processing ==> generate simple oscillator components: whats frequencies and amplitudes of a model (interprete signal processes in TensorFlow , take gradient and look at frequency alignment of input and target audio to create expressive sound): manipulate frequencies individually in selectively with knowing whats going on in the inside of the model (interpretable),  (ear (ear is sensitive to phases of sinewaves if there is no frequency): 1.) frequency decomposition)  
creativity: if only bit streams of audio (0´s and 1´s) you do not know what´s going on inside -- DB and HZ are more informative frequencies: take those for the encoder. What if I take the pitch and loudness of a different signal than the one my decoder is trained on ? 
Pitch and frequency stays the same but the tone changes according to the instrument learned by the decoder (decodes pitch and freq into e.g. a violin sound) (timbre transfer). DDSP allows to run models in real time (was never possible with raw waveform models)  

You can get very creative by trying routing lots of differents submodules in different ways. 

Training of autoencoders does not need a lot of data and usually only takes around 10 minutes: supervised by itself (todo?) 

Symbolic Representation 

Prior knowledge: how to decompose the grammar of music which works for a certain context. The philosophy is to incorporate this language and create new contexts
As a musician this gives you the opportunity to incorporate your own sounds and turn them into something new, but also for "non musician people" who like the thrill of experimenting and creating. 






  

