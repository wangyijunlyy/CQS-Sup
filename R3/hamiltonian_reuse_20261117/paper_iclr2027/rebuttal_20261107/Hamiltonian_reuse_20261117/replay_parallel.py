"""Run the unchanged chronological auditor with independent worker processes."""
import concurrent.futures
import audit

if __name__=='__main__':
 # Replays are Python-heavy. Processes avoid thread contention without changing
 # any measurement, update, retention, hash or accounting checks in audit.py.
 audit.concurrent.futures.ThreadPoolExecutor=concurrent.futures.ProcessPoolExecutor
 audit.main()
