#CUDA_VISIBLE_DEVICES=2 python -W ignore generate_speech.py \
#  --config result/TTS_result/ImageSpeechGeneration_Final/LibriTTS_1e4_4Layer_16alpha_16rank/configs_training.yaml \
#  --dataset "test" --folder2save "prediction_tts" \
#  --debug


#CUDA_VISIBLE_DEVICES=1 python -W ignore generate_speech_image.py \
#  --config result/TTS_result/ImageSpeechGeneration_Final/LibriTTS_1e4_2Layer_16alpha_16rank/configs_training.yaml \
#  --dataset "test" --folder2save "prediction_tts" \
#  --debug


CUDA_VISIBLE_DEVICES=0 python -W ignore generate_speech_cosy_image.py \
  --config result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce/LibriTTS_1e4_4Layer_16alpha_16rank_BS14_NoRemoveDup/configs_training.yaml \
  --dataset "test" --folder2save "prediction_tts" \
  --debug