#!/bin/bash
# Скрипт для проверки статуса обучения

echo "=== Статус обучения ==="
echo ""

# Проверка процессов
# Ищем основной процесс обучения
# PyTorch DataLoader с num_workers создает worker процессы для загрузки данных
# Это нормальное поведение, не ошибка!

MAIN_PROCESS=$(ps aux | grep "commands.py train" | grep -v grep | grep -E "(python|uv run)" | head -1)

if [ -z "$MAIN_PROCESS" ]; then
    echo "❌ Обучение НЕ запущено"
else
    # Подсчитываем все процессы, связанные с обучением
    ALL_PROCESSES=$(ps aux | grep "commands.py train" | grep -v grep)
    TOTAL_COUNT=$(echo "$ALL_PROCESSES" | wc -l)
    
    # Извлекаем PID основного процесса
    MAIN_PID=$(echo "$MAIN_PROCESS" | awk '{print $2}')
    
    echo "✅ Обучение запущено"
    echo "  Основной процесс:"
    echo "$MAIN_PROCESS" | awk '{print "    PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%", "| Time:", $10}'
    
    # Подсчитываем worker процессы (дочерние процессы основного Python процесса)
    # Worker процессы создаются PyTorch DataLoader для параллельной загрузки данных
    WORKER_COUNT=$(ps --ppid "$MAIN_PID" 2>/dev/null | tail -n +2 | wc -l)
    
    if [ "$WORKER_COUNT" -gt 0 ]; then
        echo "  Worker процессов (DataLoader): $WORKER_COUNT"
        echo "  Всего процессов: $TOTAL_COUNT (1 основной + ~$WORKER_COUNT workers)"
        echo "  ℹ️  Это нормально! PyTorch создает worker процессы для загрузки данных (num_workers=4)"
    else
        echo "  Всего процессов: $TOTAL_COUNT"
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


