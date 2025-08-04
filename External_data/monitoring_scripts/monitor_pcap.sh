#!/usr/bin/env bash
set -euo pipefail

# Configurazione
PCAP_DIR="/home/user/tpotce/data/suricata/log"
OUTPUT_DIR="/home/user/logs"
PROCESSED_LOG="/var/log/monitor_pcap/processed_files.log"
PYTHON_LOG="/var/log/monitor_pcap/python_output.log"
LOCK_DIR="/var/lock/monitor_pcap"
VENV="/home/user/trafficML/venv"
PYTHON_SCRIPT="/home/user/trafficML/main.py"
REMOTE_USER="root"
REMOTE_HOST="0.0.0.0"
REMOTE_DIR="/root/attack_logs"
INOTIFY_EVENTS="-m -e create --format '%f'"

mkdir -p "$(dirname "$PROCESSED_LOG")" "$(dirname "$PYTHON_LOG")" "$LOCK_DIR"
touch "$PROCESSED_LOG" "$PYTHON_LOG"

log() {
  local LEVEL="$1" MSG="$2"
  echo "$(date +'%F %T') [$LEVEL] $MSG" | tee -a "$PYTHON_LOG"
}

process_file() {
  local FILENAME="$1"
  local FILEPATH="$PCAP_DIR/$FILENAME"
  local BASENAME="${FILENAME%.pcap}"
  local OUTPUT_SUB="$OUTPUT_DIR/$BASENAME"
  local OUTPUT_CSV="$OUTPUT_SUB/${BASENAME}_Flow.csv"

  # Lock per file
  exec 200>"$LOCK_DIR/proc_${BASENAME}.lock"
  flock -n 200 || {
    log "WARN" "Lock attivo per $BASENAME, skip."
    return
  }

  log "INFO" "Inizio elaborazione di $FILEPATH"
  cfm "$FILEPATH" "$OUTPUT_DIR" || {
    log "ERROR" "Errore cfm su $FILEPATH"; return
  }

  if [[ ! -f "$OUTPUT_CSV" ]]; then
    log "ERROR" "CSV non generato in $OUTPUT_CSV"; return
  fi

  if [[ ! -f "$VENV/bin/activate" ]]; then
    log "ERROR" "Virtualenv non trovato in $VENV"; return
  fi

  # Esegui script Python
  log "INFO" "Attivo venv e lancio script su $OUTPUT_CSV"
  source "$VENV/bin/activate"
  python3 "$PYTHON_SCRIPT" "$OUTPUT_CSV" >>"$PYTHON_LOG" 2>&1 || {
    log "ERROR" "Fallito script Python su $OUTPUT_CSV"; deactivate; return
  }
  deactivate

  echo "$FILEPATH" >>"$PROCESSED_LOG"
  log "INFO" "Script Python eseguito, file processato registrato."

  transfer_logs
}

transfer_logs() {
  local LOG_DIR="$(dirname "$PYTHON_SCRIPT")/out/logs"
  if [[ -d "$LOG_DIR" ]]; then
    for file in "$LOG_DIR"/*; do
      [[ -f "$file" ]] || continue
      log "INFO" "Trasferimento di $file"
      local attempts=0
      until (( attempts >= 3 )); do
        scp -o ConnectTimeout=10 "$file" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" && break
        ((attempts++))
        log "WARN" "Tentativo $attempts fallito per $file"
        sleep 2
      done
      if [[ -f "$file" ]]; then
        rm "$file" && log "INFO" "Rimosso $file dopo trasferimento"
      fi
    done
  else
    log "WARN" "Nessuna directory di logs '$LOG_DIR' trovata"
  fi
}

monitor() {
  while true; do
    inotifywait $INOTIFY_EVENTS "$PCAP_DIR" 2>>"$PYTHON_LOG" | while read FILENAME; do
      [[ "$FILENAME" == *.pcap ]] || continue
      grep -Fxq "$PCAP_DIR/$FILENAME" "$PROCESSED_LOG" || process_file "$FILENAME"
    done
    log "ERROR" "inotifywait terminato, riavvio monitoraggio dopo 5s"
    sleep 5
  done
}

# Avvio 
log "INFO" "Avvio monitor directory: $PCAP_DIR"
monitor