from psychopy import parallel
import pandas as pd

pp = parallel.ParallelPort(0)
pp.port.setDataDir(0)

def collect_pp_data():
    store = [] 
    try:
        while True:
            ack = pp.port.getInAcknowledge()
            busy = pp.port.getInBusy()
            paper = pp.port.getInPaperOut()
            select = pp.port.getInSelected()
            error = pp.port.getInError()
            data = pp.port.getData()
            data_dir = pp.port.dataDir()
            col = (ack, busy, paper, select, error, data, data_dir)
            store.append(col)
    except KeyboardInterrupt:
        df = pd.DataFrame(store, columns=["ack","busy","paper","select","error","data","data_dir"])
        df.to_csv("pp_data_2.csv")
