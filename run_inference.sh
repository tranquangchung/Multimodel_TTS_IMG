#CUDA_VISIBLE_DEVICES=2 python -W ignore generate_speech.py \
#  --config result/TTS_result/ImageSpeechGeneration_Final/LibriTTS_1e4_4Layer_16alpha_16rank/configs_training.yaml \
#  --dataset "test" --folder2save "prediction_tts" \
#  --debug


#CUDA_VISIBLE_DEVICES=1 python -W ignore generate_speech_image.py \
#  --config result/TTS_result/ImageSpeechGeneration_Final/LibriTTS_1e4_2Layer_16alpha_16rank/configs_training.yaml \
#  --dataset "test" --folder2save "prediction_tts" \
#  --debug


#CUDA_VISIBLE_DEVICES=0 python -W ignore generate_speech_cosy_image.py \
#  --config result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce/LibriTTS_1e4_4Layer_16alpha_16rank_BS14_RemoveDup_KeepPunctuation/configs_training.yaml \
#  --debug

#CUDA_VISIBLE_DEVICES=0 python -W ignore generate_speech_cosy.py \
#  --config result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce_PromptV2/LibriTTS_1e4_8Layer_16alpha_16rank_BS14_RemoveDup_KeepPunctuation/configs_training.yaml \
#  --dataset "test-clean" --folder2save "prediction_tts" \

#CUDA_VISIBLE_DEVICES=2 python -W ignore generate_speech_cosy_prompt.py \
#  --config result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce_PromptV2/EARSx5_Libris960_1e4_8Layer_16alpha_16rank_BS14_CLIP_fintuneLibris/configs_training.yaml \
#  --dataset "test-clean" --folder2save "prediction_tts_clip" \

#CUDA_VISIBLE_DEVICES=3 python -W ignore generate_speech_cosy_prompt_t5.py \
#  --config result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce_PromptV2/EARSx5_Libris960_1e4_8Layer_16alpha_16rank_BS14_T5_fintuneLibris/configs_training.yaml \
#  --dataset "test-clean" --folder2save "prediction_tts_t5" \
