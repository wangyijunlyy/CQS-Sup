"""Stop only experiment-owned subprocess objects after monitoring failure."""
import subprocess

def stop_owned(jobs):
 surviving=[p for _,p in jobs if p.poll() is None]
 for p in surviving:p.terminate()
 for p in surviving:
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
