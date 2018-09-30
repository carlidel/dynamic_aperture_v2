# dynamic_aperture_v2

## Brief indication of the repository

### condor
It just contains the code I used for generating the Hennon map data. It's a .cpp file that takes a lot of arguments and then sent to HTCondor with the file ang_job.sub with the enormous list of paramters in the .txt file. All the outputs are then merged toghether with a python script into a dictionary (you'll see that I really love dictionaries, at least for this project).

### LHC
It contains the data made by the symplectic traking of the LHC

### MORE_DATA
It contains some more data of the symplectic traking but it has been already analysed with other methods, this is interesting because it can be a good comparison point for our methods.

### the scripts all around
90% of the work was done on ```fit_library.py``` and ```script_fit.py```. I suggest you to use Spyder as your main editor since, as you can see, everything was splitted with ```#%%``` in ```script_fit.py```.

If anything in my code is not clear (let's be sincere here, it's not an "if", it's a "when", this code is a mess in so many points) don't hesitate to contact me.

Good luck and stay strong, comrade!