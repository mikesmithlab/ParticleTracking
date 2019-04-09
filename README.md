# ParticleTracking

This repository is used for locating and tracking a variety of particles and then for subsequent analysis of their motion.

- annotation --> For adding annotations to video files
    - Class VideoAnnotator
- dataframes --> Managing DataStores containing dataframes to record data
    - particle_data : contains information for every particle in every frame
    - frame_data : contains ensemble data for each frame
- graphs --> Contains classes to plot graphs of results. - Working progress - Contains bad functions.
- tracking --> Tracks particles in a video
- preprocessing --> Processes each frame before tracking takes place 
- statistics --> Functions to perform common analysis on the tracking data
