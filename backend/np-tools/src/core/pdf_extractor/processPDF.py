import argparse
import time 

from datetime import datetime
import json
import time

#Propias:
from src.MultithreadProcess import processPDF



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', help='Input directory name', required=True)
    parser.add_argument('-o', '--outdir', help='Output directory name', required=True)
    parser.add_argument('-w', '--workers', help='number of workers', default=4)
    args = parser.parse_args()    


    #data = processPDF.processPDFSimple ('/export/data_ml4ds/thuban/repository/data/Articulos2/26.pdf')
    #import ipdb ; ipdb.set_trace()
    #exit()

    process = processPDF (args.indir, args.outdir, int(args.workers))

    now = datetime.now()
    start_time = time.time()
    data = process.processFiles ()
    end_time = time.time()
    execution_time = end_time - start_time

    logdata ={'datetime':now.strftime('%Y-%m-%d'),
              'executiontime':execution_time //60,
              'inputdir': args.indir,
              'numerfiles': len (data),
              'failedfiles':len ([d for d in data if d['result']==False]),
              'listfiles': data
              }
    with open('/tmp/log.json', 'w') as fp:
        json.dump(logdata, fp)






