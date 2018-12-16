# Visual Rhythm Detection

## Run

```
python train.py \
    --audio_dir ../audio_3_split \
    --video_dir ../video_3_frames_4fps \
    --debug 1 \
    --device 0 \
    --eval_every 1 \
    --fps 4 \
    --vgg_init 1 \
    --use_crf 1 \
    --use_label 1 \
    --model EndToEnd
```
