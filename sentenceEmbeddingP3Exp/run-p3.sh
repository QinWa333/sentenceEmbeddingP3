#!/bin/bash

# P3-STS评估脚本
# 该脚本运行P3-STS方法进行句子嵌入评估

# 默认配置
CONFIG_NAME="llama-2-7b-p3"
CONFIG_FILE="config-p3.yaml"
LOG_FILE="p3.log"
RUN_IN_BACKGROUND=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --num_placeholders)
            NUM_PLACEHOLDERS="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        --bg|--background)
            RUN_IN_BACKGROUND=true
            shift
            ;;
        *)
            echo "未知选项: $1"
            echo "用法: $0 [--config CONFIG_NAME] [--config_file CONFIG_FILE] [--num_placeholders NUM] [--log LOG_FILE] [--bg]"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python -u evaluate-p3.py --config $CONFIG_NAME --config_file $CONFIG_FILE"

if [ ! -z "$NUM_PLACEHOLDERS" ]; then
    CMD="$CMD --num_placeholders $NUM_PLACEHOLDERS"
fi

echo "使用配置运行P3-STS评估: $CONFIG_NAME"
echo "日志文件: $LOG_FILE"
echo "命令: $CMD"

if [ "$RUN_IN_BACKGROUND" = true ]; then
    echo "在后台运行进程..."
    echo "使用 'tail -f $LOG_FILE' 查看实时日志"
    # 修正：使用正确的 nohup 命令
    nohup $CMD > "$LOG_FILE" 2>&1 &
    echo "进程PID: $!"
else
    echo "在前台运行进程"
    echo "按 Ctrl+C 停止进程"
    # 前台运行不需要 nohup，使用 tee 可以同时看到输出
    $CMD 2>&1 | tee "$LOG_FILE"
fi