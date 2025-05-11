# BabyLM-Challenge
Team NA1 Model implementation for the BabyLM Challenge.

GPT2 decoder model with entropy and adaptive curriculum based design.

Prerequisites:
  - Remeber if using hpc scripts to change ur user name.
  - For cuda compatibility check working torch version compatible with cuda (https://discuss.pytorch.org/t/which-cuda-version-to-choose-when-installing-pytorch/217257)

Local machine excercution:
- If running locally, parse args are required.
- Parse arguments for script are as follows:

      param_tuning.py  '--data_path tokenizers/10M_data_token.pkl --toggle_scheduler off --score_type Loss --n_tokens 500_000 --proxy_n_trials 2 --main_n_trials 3 --proxy_train on --main_train off' (inspect file for arg options)
  
      gpt_model.py     '--data_path tokenizers/10M_data_token.pkl --scoring composite --schedule_type linear --curriculum on --data_size None ' (inspect file for arg options)
  
      proxy_main.py    '--data_path tokenizers/10M_data_token.pkl --data_size None' (inspect file for arg options)
  
- Models and data will be returned to the trained_models directory. Check script for specific saving mechanics.

HPC Usage:
- Hpc scripts will have to be adjusted by user manually for sbatch. User name corrections must be applied for user specific.
- All error and output files will be found in the 'logs' directory.
- GPU usage can be count in the .out files for each model. Check gpu is being used (indicated by '1MiB /  81920MiB |      0%'), if idle check pytorch version compatibility, if still not working with compatibile torch correct configs to force to GPU
      
