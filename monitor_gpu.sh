#!/bin/bash
# GPU monitoring script - run in separate terminal while training

echo "Monitoring GPU utilization..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    date
    echo "================================"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s\n  Memory: %s/%s MB (%.1f%%)\n  Utilization: %s%% GPU, %s%% Mem\n  Temp: %s°C\n", $1, $2, $3, $4, ($3/$4)*100, $5, $6, $7}'
    echo "================================"
    sleep 2
done
