

import argparse
from pathlib import Path
import queue
import threading


args = argparse.ArgumentParser()
args.add_argument('input', type=str, help='Input file')
args.add_argument('output', type=str, help='Output folder')
args.add_argument('num_threads', type=int, help='Number of threads')
args.add_argument('n', type=int, default=10000, help='Number of writes')


args = args.parse_args()

q = queue.Queue()

def worker():
    while True:
      item = q.get()

      if item is None:
        break

      data, filename = q.get()
      
      # write file binary
      with open(filename, 'wb') as f:
          f.write(data)
    

# read file binary
with open(args.input, 'rb') as f:
    data = f.read()


threads = [
    threading.Thread(target=worker) for _ in range(args.num_threads)
]

for t in threads:
    t.start()


output = Path(args.output)
output.mkdir(exist_ok=True, parents=True)

