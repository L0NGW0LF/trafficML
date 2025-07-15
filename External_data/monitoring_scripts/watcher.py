#!/opt/attack_pipeline/venv/bin/python
import time
import json
import csv
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configurazione 
WATCH_DIR = "/root/attack_logs"
OUTPUT_FILE = "/var/log/attack_logs/attack_logs.json"
PROCESSED_DIR = os.path.join(WATCH_DIR, "processed")
LOG_FILE = "/var/log/attack_pipeline_watcher.log" # File di log per lo script stesso

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() 
    ]
)

class AtomicMoveHandler(FileSystemEventHandler):

    def process(self, event):
        # Ignora le directory e i file temporanei (es. .tmp)
        if event.is_directory or event.src_path.endswith('.tmp'):
            return

        # Processa solo file con estensione .csv
        if event.src_path.endswith('.csv'):
            logging.info(f"Rilevato file CSV valido: {event.src_path}")
            process_csv(event.src_path)

    def on_created(self, event):
        # Delay per garantire che il file sia completamente scritto
        time.sleep(1)
        self.process(event)

    def on_moved(self, event):
        # Gestione del file spostato
        logging.info(f"File spostato nella directory di monitoraggio: {event.dest_path}")
        self.process(type('obj', (object,), {'is_directory': event.is_directory, 'src_path': event.dest_path})())


def process_csv(path):
    # Lettura CSV attacchi
    
    if not os.path.exists(path):
        logging.warning(f"Il file {path} è scomparso prima di poter essere processato. Ignoro.")
        return

    try:
        processed_count = 0
        skipped_count = 0
        with open(path, 'r', encoding='utf-8') as csvfile, open(OUTPUT_FILE, 'a') as logfile:
            reader = csv.reader(csvfile)

            # Logica per saltare la riga di intestazione
            try:
                header = next(reader)
                logging.info(f"Intestazione saltata per il file {path}: {header}")
            except StopIteration:
                logging.warning(f"File {path} è vuoto, nessuna riga da processare.")
            else:
                # Lettura righe di dati
                for i, row in enumerate(reader, start=1):
                    try:
                        if len(row) != 4:
                            logging.warning(f"Riga {i} saltata in {path}: numero di colonne errato ({len(row)}). Contenuto: {row}")
                            skipped_count += 1
                            continue

                        src, dst, port, label = [field.strip() for field in row]

                        if not all([src, dst, port, label]):
                            logging.warning(f"Riga {i} saltata in {path}: uno o più campi sono vuoti. Contenuto: {row}")
                            skipped_count += 1
                            continue

                        event = {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
                            "log_type": "ml_attack_log",
                            "srcip": src,
                            "dstip": dst,
                            "dstport": port,
                            "predicted_label": label
                        }
                        logfile.write(json.dumps(event) + "\n")
                        processed_count += 1
                    except Exception as e:
                        logging.error(f"Errore durante l'elaborazione della riga {i} in {path}: {e}. Riga: {row}")
                        skipped_count += 1

            logging.info(f"Elaborazione di {path} completata. Righe processate: {processed_count}, Righe saltate: {skipped_count}")

    except FileNotFoundError:
        logging.error(f"File non trovato durante l'elaborazione: {path}.")
        return
    except Exception as e:
        logging.critical(f"Errore critico durante l'elaborazione del file {path}: {e}")
        return

    # Spostamento del file nella directory di archivio
    try:
        filename = os.path.basename(path)
        destination_path = os.path.join(PROCESSED_DIR, filename)

        if os.path.exists(destination_path):
            timestamp_suffix = f"_{int(time.time())}"
            name, ext = os.path.splitext(filename)
            destination_path = os.path.join(PROCESSED_DIR, f"{name}{timestamp_suffix}{ext}")

        os.rename(path, destination_path)
        logging.info(f"File spostato in archivio: {destination_path}")
    except Exception as e:
        logging.error(f"Impossibile spostare il file {path} in archivio: {e}")

# Funzione principale per avviare il servizio di monitoraggio
def main():
    
    logging.info("Avvio del servizio di monitoraggio file...")

    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        logging.info(f"Directory di monitoraggio: {WATCH_DIR}")
        logging.info(f"File di output JSON: {OUTPUT_FILE}")
        logging.info(f"Directory di archivio: {PROCESSED_DIR}")
    except OSError as e:
        logging.critical(f"Impossibile creare le directory necessarie: {e}")
        return

    observer = Observer()
    event_handler = AtomicMoveHandler()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    logging.info("Servizio avviato e in attesa di file...")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Rilevato arresto manuale (Ctrl+C).")
    finally:
        logging.info("Arresto del servizio di monitoraggio...")
        observer.stop()
        observer.join()
        logging.info("Servizio arrestato correttamente.")

if __name__ == "__main__":
    main()