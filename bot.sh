#!/bin/bash
# Управление ботом GoTrips RAG
# Использование: ./bot.sh start | stop | restart | status | logs

PIDFILE="/home/user/rag_gotrips/bot.pid"
LOGFILE="/home/user/rag_gotrips/bot.log"
SCRIPT="/home/user/rag_gotrips/3_bot_telegram.py"
ENVFILE="/home/user/rag_gotrips/.env"

is_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

do_stop() {
    # Убиваем по PID-файлу
    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Останавливаю PID $pid..."
            kill "$pid"
            sleep 3
            kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$PIDFILE"
    fi
    # Убиваем любые оставшиеся процессы бота
    pkill -f "python3.*3_bot_telegram" 2>/dev/null && echo "Убиты зависшие процессы" || true
    sleep 1
}

do_start() {
    if is_running; then
        echo "Бот уже запущен (PID $(cat $PIDFILE))"
        exit 1
    fi
    do_stop  # на всякий случай зачистить

    export $(grep -v '^#' "$ENVFILE" | xargs)

    nohup python3 "$SCRIPT" >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    echo "Бот запущен (PID $!)"
    echo "Лог: tail -f $LOGFILE"
}

case "$1" in
    start)
        do_start
        ;;
    stop)
        do_stop
        echo "Бот остановлен."
        ;;
    restart)
        do_stop
        sleep 2
        do_start
        ;;
    status)
        if is_running; then
            echo "Бот РАБОТАЕТ (PID $(cat $PIDFILE))"
            echo "Памяти:"
            ps -p "$(cat $PIDFILE)" -o pid,%mem,rss,vsz --no-headers 2>/dev/null || true
        else
            echo "Бот НЕ запущен"
        fi
        ;;
    logs)
        tail -f "$LOGFILE"
        ;;
    *)
        echo "Использование: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
