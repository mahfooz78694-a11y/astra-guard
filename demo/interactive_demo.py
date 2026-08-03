import time
import sys

def simulate_realtime_log():
    logs = [
        "[SYSTEM] Initializing ASTRA-GUARD Engine...",
        "[SYSTEM] Loading Pre-trained Vision Model...",
        "[ASTRA] Scanning architecture for optimal bottleneck layer...",
        "[ASTRA] Found candidate layer: 'layer4.2.relu'",
        "[ASTRA] Calibrating guard profile on uncorrupted dataloader...",
        "[VORTEX] Subspace locked. Matrix detached from autograd graph.",
        "[SYSTEM] Live Inference Started on port 8080.\n",
        "--- INCOMING REQUEST STREAM ---",
        "[LIVE] Request ID: 001 | Tensor Shape: [32, 3, 224, 224] | Status: CLEAN",
        "[LIVE] Request ID: 002 | Tensor Shape: [32, 3, 224, 224] | Status: CLEAN",
        "\n!!! WARNING: MALICIOUS PAYLOAD DETECTED !!!",
        "[ALERT] Request ID: 003 | Attack Signature: PGD-100 L-inf",
        "[INTERCEPTOR] Triggering Layer-7 Forward Hook...",
        "[INTERCEPTOR] Intercepted intermediate activation at 'layer4.2.relu'.",
        "[VORTEX] Processing transient tensor...",
        "[VORTEX] Applying proprietary defense mechanism...",
        "[VORTEX] Neutralizing adversarial perturbation...",
        "[INTERCEPTOR] Restoring sanitized tensor.",
        "[INTERCEPTOR] Overwriting forward pass with sanitized activation.",
        "[SUCCESS] Adversarial perturbation neutralized. Model confidence restored.",
        "\n--- RESUMING NORMAL OPERATION ---",
        "[LIVE] Request ID: 004 | Tensor Shape: [32, 3, 224, 224] | Status: CLEAN"
    ]

    print("\n========================================================")
    print("  ASTRA-GUARD: Zero-Retraining AI Activation Firewall  ")
    print("========================================================\n")

    for log in logs:
        print(log)
        # Flush to mimic real-time output
        sys.stdout.flush()
        # Add varied delay based on message type
        if "WARNING" in log or "ALERT" in log:
            time.sleep(0.8)
        elif "INTERCEPTOR" in log or "VORTEX" in log:
            time.sleep(0.4)
        else:
            time.sleep(0.2)

    print("\n[SYSTEM] Demo completed successfully.\n")

if __name__ == "__main__":
    try:
        simulate_realtime_log()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Demo terminated by user.")
