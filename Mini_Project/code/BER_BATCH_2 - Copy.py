from BER_CSV_QPSK import main as QPSK
from BER_CSV_8PSK import main as _8PSK
from BER_CSV_16PSK import main as _16PSK
import multiprocessing as mp


if __name__ == '__main__':
    processes = [mp.Process(target=QPSK), mp.Process(target=_8PSK), mp.Process(target=_16PSK)]
    for p in processes:
        p.start()
    
        
        