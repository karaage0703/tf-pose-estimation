#!/bin/sh
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
python3 skeleton_sequencer.py -d='jetson_nano_raspi_cam' --fullscreen='True'
