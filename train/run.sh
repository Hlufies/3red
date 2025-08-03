#!/bin/bash

# 正确设置环境变量（去除空格）
export RUNTIME_SCRIPT_DIR="/Users/huang/Desktop/TX/v1"
export TRAIN_LOG_PATH="${RUNTIME_SCRIPT_DIR}/train_log"
export TRAIN_TF_EVENTS_PATH="${TRAIN_LOG_PATH}/tf_events"
export TRAIN_DATA_PATH="/Users/huang/Desktop/TX/dataset/TencentGR_1k"
export TRAIN_CKPT_PATH="${TRAIN_LOG_PATH}/ckpt"

# 验证环境变量是否设置成功
echo "RUNTIME_SCRIPT_DIR: $RUNTIME_SCRIPT_DIR"
echo "TRAIN_LOG_PATH: $TRAIN_LOG_PATH"


# 进入工作目录
cd "${RUNTIME_SCRIPT_DIR}"

# 执行Python程序
python -u main.py