using the module to train CNN model by running \n
```
python3 -W ignore model_training/sentiment_training.py \
--train /home/jovyan/work/E4577-AI-OR-at-Scale-in-the-Cloud/dataset/train/ \
--validation /home/jovyan/work/E4577-AI-OR-at-Scale-in-the-Cloud/dataset/dev/ \
--eval /home/jovyan/work/E4577-AI-OR-at-Scale-in-the-Cloud/dataset/eval/ \
--model_output_dir /home/jovyan/work/E4577-AI-OR-at-Scale-in-the-Cloud/model_output/ \
--num_epoch 20 \
--config_file /home/jovyan/work/E4577-AI-OR-at-Scale-in-the-Cloud/model_training/training_config.json
```