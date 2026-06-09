"""Kill old api_server and start fresh one, then run batch evaluation."""
import os, sys, signal, time, subprocess

# Find and kill old api_server holding port 8081
try:
    import psutil
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'api_server' in cmdline and proc.info['pid'] != os.getpid():
                print(f"Killing old api_server PID={proc.info['pid']}")
                proc.kill()
                time.sleep(2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
except ImportError:
    # fallback: use lsof
    import subprocess
    result = subprocess.run(['lsof', '-ti', ':8081'], capture_output=True, text=True)
    for pid in result.stdout.strip().split('\n'):
        if pid:
            print(f"Killing PID {pid}")
            os.kill(int(pid), signal.SIGKILL)
    time.sleep(2)

print("Starting new api_server...")
srv = subprocess.Popen(
    [sys.executable, 'api_server.py'],
    cwd=os.path.dirname(os.path.abspath(__file__)),
)
print(f"api_server PID={srv.pid}")

# Wait for health
import urllib.request, json
for i in range(60):
    time.sleep(3)
    try:
        r = urllib.request.urlopen('http://localhost:8081/api/v1/health', timeout=3)
        data = json.loads(r.read())
        if data.get('model_loaded'):
            print(f"Server ready after {(i+1)*3}s")
            break
    except Exception:
        print(f"  waiting... ({(i+1)*3}s)")
else:
    print("Server did not start in time, exiting")
    srv.terminate()
    sys.exit(1)

print("\nStarting batch evaluation...")
result = subprocess.run([sys.executable, 'batch_evaluate.py'],
                        cwd=os.path.dirname(os.path.abspath(__file__)))
sys.exit(result.returncode)
