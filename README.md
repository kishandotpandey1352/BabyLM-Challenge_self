# BabyLM-Challenge
Team NA1 Model implementation for the BabyLM Challenge.

GPT2 decoder model with entropy and adaptive curriculum based design.

Prerequisites:
  - Remeber if using hpc scripts to change your user name.
  - For cuda compatibility check that the torch version you have is compatible with cuda (https://discuss.pytorch.org/t/which-cuda-version-to-choose-when-installing-pytorch/217257)

Local machine excercution:
- If running locally, parse args are required.
- Parse arguments for script are as follows:

      param_tuning.py  '--data_path tokenizers/10M_data_token.pkl --toggle_scheduler off --score_type Loss --n_tokens 500_000 --proxy_n_trials 2 --main_n_trials 3 --proxy_train on --main_train off' (inspect file for arg options)
  
      gpt_model.py     '--data_path tokenizers/10M_data_token.pkl --scoring composite --schedule_type linear --curriculum on --data_size None ' (inspect file for arg options)
  
      proxy_main.py    '--data_path tokenizers/10M_data_token.pkl --data_size None' (inspect file for arg options)
  
- Models and data will be returned to the trained_models directory. Check script for specific saving mechanics.

HPC Usage:
- Hpc scripts will have to be adjusted by user manually for sbatch. User name corrections must be applied for user specific username on stanage.
- All error and output files will be found in the 'logs' directory.
- GPU usage can be count in the .out files for each model. Check gpu is being used (indicated by '1MiB /  81920MiB |      0%'), if idle check pytorch version compatibility which can be seen by passing 'conda list' to the login node on Stanage, if still not working with compatibile torch correct configs to force to GPU (device = torch.device("cuda"), and check pytorch-mutex is not installed or it will overwrite pytorch-cuda.
- Create a conda enviroment with compatible packages (numpy, torch, sklearn, optuna, pandas) called babylm, or edit the activate command in bash files to you preferred enviroment. 

General Instructions for Local Use:
- scripts in the main and hps file can be run locally.
- In order to run locally see the above 'local machine excercution' section above. The arguments avalible for each argument are as follows:
  
              param_tuning.py accepts args:
                 --data_path   -> this directs the model to train ok specified data. In this repo it is found inside tokenisers, e.g 'tokenizers/10M_data_token.pkl'
                 --toggle_scheduler  -> this controls if the curriculum scheduling is on or off to allow for basline and enhanced training, arguments are ['on','off']
                 --score_type  -> this controls the type of scorring returned by the proxy trainer, that i then passed to the scheduler, arguments are ['composite','Loss','Entropy']
                 --n_tokens  -> this controls the number of tokens exposed to the model (tokens are shuffled before passing so do not have any order), for testing on and hyper param tuning this should be specified to increase train speed, when training full data pass ['None'] else specify token count, e.g 500_000
                 --proxy_n_trials  -> this specifies the number of trials run by optuna, a baysian hp tuning library, it accpets any non negative or 0 value.
                 --main_n_trials  -> same applies as proxy_n_trials, recommended use more for full model tuning.

              gpt_model.py accepts args:
                 --data_path  -> direscts to training data for model
                 --scoring  -> this controls the type of scoring by the proxy model, arguments accepted are ['composite','Loss','Entropy']
                 --schedule_type  -> this controls the type of curriculum the sceduler will produces, options are ['linear','tanh','sigmoid','log','linear']
                 --curriculum  -> this toggles the curriculum on and off to allow for baseline and enhanced model training, options are ['on','off']
                 --data_size -> data size specifies the number of tokens seen by the data and file output name, accepts None and any integer value

              proxy main arguments:
                 --data_pathq  -> specifies data path used for training
                 --data_size   -> this controls how many tokens the model is exposed to, accepts None and and integer value.
  
