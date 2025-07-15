#!/usr/bin/env python3
import sys, json, subprocess
import datetime

# Define the path for the Active Response log file
log_file_path = "/var/ossec/logs/active-responses.log"

def log_action(log_data):
    """Writes a log action to the specified file."""
    try:
        # Adds timestamp and script name to the log data
        log_data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_data["ar_script"] = "block_ip.py"

        with open(log_file_path, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    except Exception as e:
        print(f"Error writing to log file {log_file_path}: {e}", file=sys.stderr)

# --- START MAIN LOGIC ---

try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    log_action({"status": "error", "message": "Invalid JSON input"})
    sys.exit(1)

command = data.get("command")

# --- Improved data extraction ---
srcip = None
dstip = None # NUOVA VARIABILE
rule_id = None
attack_type = "Unknown" # Default value

alert_data = data.get("parameters", {}).get("alert", {})
if "data" in alert_data:
    # The srcip, dstip and predicted_label fields are in the 'data' dictionary of the alert
    srcip = alert_data["data"].get("srcip")
    dstip = alert_data["data"].get("dstip", "N/A") # ESTRAE L'IP DI DESTINAZIONE
    attack_type = alert_data["data"].get("predicted_label", "Unknown")

if "rule" in alert_data:
    rule_id = alert_data["rule"].get("id")

if not srcip:
    log_action({"status": "error", "message": "Source IP not found in alert data", "alert_data": alert_data})
    sys.exit(1)


# Base dictionary for logging, to be updated based on the result
log_payload = {
    "ip_address": srcip,
    "destination_ip": dstip, # AGGIUNTO L'IP DI DESTINAZIONE AL PAYLOAD
    "triggering_rule": rule_id,
    "attack_type": attack_type
}

if command == "add":
    log_payload["ar_command"] = "add"
    try:
        result = subprocess.run(["iptables", "-A", "INPUT", "-s", srcip, "-j", "DROP"], capture_output=True, text=True, check=True)
        log_payload["status"] = "success"
        log_payload["message"] = f"Blocked IP {srcip} for attacking {dstip}. Reason: {attack_type} attack"
        log_action(log_payload)
    except subprocess.CalledProcessError as e:
        log_payload["status"] = "failure"
        log_payload["error"] = e.stderr.strip()
        log_payload["message"] = f"Failed to block IP {srcip}"
        log_action(log_payload)
    except Exception as e:
        log_payload["status"] = "error"
        log_payload["error"] = str(e)
        log_payload["message"] = f"Exception during block attempt for {srcip}"
        log_action(log_payload)

elif command == "delete":
    log_payload["ar_command"] = "delete"
    try:
        result = subprocess.run(["iptables", "-D", "INPUT", "-s", srcip, "-j", "DROP"], capture_output=True, text=True, check=True)
        log_payload["status"] = "success"
        log_payload["message"] = f"Unblocked IP {srcip}"
        log_action(log_payload)
    except subprocess.CalledProcessError as e:
        log_payload["status"] = "failure"
        log_payload["error"] = e.stderr.strip()
        log_payload["message"] = f"Failed to unblock IP {srcip}"
        log_action(log_payload)
    except Exception as e:
        log_payload["status"] = "error"
        log_payload["error"] = str(e)
        log_payload["message"] = f"Exception during unblock attempt for {srcip}"
        log_action(log_payload)

sys.exit(0)