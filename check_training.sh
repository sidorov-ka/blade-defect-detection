#!/bin/bash
# Скрипт для проверки статуса обучения

echo "=== Статус обучения ==="
echo ""

# Проверка процессов
PROCESSES=$(ps aux | grep "commands.py train" | grep -v grep)
if [ -z "$PROCESSES" ]; then
    echo "❌ Обучение НЕ запущено"
else
    COUNT=$(echo "$PROCESSES" | wc -l)
    if [ "$COUNT" -eq 1 ]; then
        echo "✅ Обучение запущено (1 процесс)"
        echo "$PROCESSES" | awk '{print "  PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%", "| Time:", $10}'
    else
        echo "⚠️  Запущено $COUNT процессов обучения (должен быть 1!)"
        echo "$PROCESSES" | awk '{print "  PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%"}'
    fi
fi

echo ""

# Проверка GPU
echo "=== Использование GPU ==="
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -i python || echo "")
if [ -z "$GPU_PROCESSES" ]; then
    echo "⚠️  GPU не используется процессами Python"
else
    echo "✅ GPU используется:"
    echo "$GPU_PROCESSES"
fi

echo ""

# Проверка последнего лога
echo "=== Последний лог TensorBoard ==="
LATEST_LOG=$(ls -t lightning_logs/blade_defect_detection/version_*/events.out.tfevents* 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    SIZE=$(stat -c "%s" "$LATEST_LOG" 2>/dev/null)
    MTIME=$(stat -c "%y" "$LATEST_LOG" 2>/dev/null | cut -d'.' -f1)
    echo "Файл: $(basename $(dirname $LATEST_LOG))/$(basename $LATEST_LOG)"
    echo "Размер: $SIZE байт"
    echo "Последнее обновление: $MTIME"
    
    # Проверить, обновлялся ли недавно (в последние 30 секунд)
    NOW=$(date +%s)
    FILE_TIME=$(stat -c "%Y" "$LATEST_LOG" 2>/dev/null)
    if [ -n "$FILE_TIME" ]; then
        DIFF=$((NOW - FILE_TIME))
        if [ "$DIFF" -lt 30 ]; then
            echo "✅ Лог обновляется (обновлен $DIFF сек назад)"
        else
            echo "⚠️  Лог не обновлялся $DIFF секунд"
        fi
    fi
else
    echo "❌ Логи не найдены"
fi

echo ""

# Проверка чекпоинтов
echo "=== Чекпоинты ==="
CHECKPOINTS=$(find lightning_logs -name "*.ckpt" -type f 2>/dev/null | wc -l)
if [ "$CHECKPOINTS" -gt 0 ]; then
    echo "✅ Найдено чекпоинтов: $CHECKPOINTS"
    find lightning_logs -name "*.ckpt" -type f 2>/dev/null | head -3 | while read ckpt; do
        echo "  - $(basename $(dirname $ckpt))/$(basename $ckpt)"
    done
else
    echo "⚠️  Чекпоинты не найдены"
fi

