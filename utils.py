from datetime import datetime

def prints(message, log_file="train_ensemble_log.txt"):
    # Print to screen (no timestamp)
    print(message)

    # Format message with timestamp for log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")
