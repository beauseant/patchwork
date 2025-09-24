import glob
import json
import argparse
import os
import pandas as pd
import language_tool_python
from src.utils import clean_string

def getDataFiles (path):
    return glob.glob( path + '/**/*.json', recursive=True)

def getPages (pages):
    return [ (page['element_content']) for page in  pages ]

def getSummary (filename):
    try:
        with open( filename ) as user_file:
            return user_file.read()
    except:
        return ""
def correctText (text):
    lang_tool = language_tool_python.LanguageTool('es-ES', remote_server='http://0.0.0.0:8081')
    try:
        textc = lang_tool.correct (text)
    except:
        textc = 'NA'
        import ipdb;ipdb.set_trace()

    return textc

def getFileData (file, tool):
    try:
        output = {}
        with open( file ) as user_file:
            data = json.load (user_file)
            output['metadata'] = data['metadata']
            output['raw_text'] = " ".join (str(element[0]) for element in [ getPages (page['content']) for page in data['pages']])
            #textc = clean_string (output['raw_text']).encode().decode('UTF-8', errors='ignore').split()
            textc = clean_string (output['raw_text']).split()
            num = 1000
            chunks = [textc[i:i + num] for i in range(0, len(textc), num)]
            #lang_tool = language_tool_python.LanguageTool('en-US', remote_server='http://0.0.0.0:8081')
            #import ipdb ; ipdb.set_trace()
            #textc = lang_tool.correct (text)
            chunksc = [ correctText (' '.join (chunk)) for chunk in chunks]
            #import ipdb ; ipdb.set_trace()
            output['raw_textc'] =  ' '.join(str(chunk) for chunk in chunksc)
            output['pdf_path'] = data['metadata']['file_path']
            output['json_path'] = os.path.dirname(file)
            output['summary'] = getSummary (os.path.join (output['json_path'], 'summary.txt'))

    except Exception as E:
        print (str(E))
        import ipdb ; ipdb.set_trace()

    print (os.path.basename(file))
    return output    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', help='Input directory name', required=True)
    parser.add_argument('-p', '--outfile', help='parquet file to save', required=True)
    parser.add_argument('-e',action='store_true',help='Does not delete documents with a blank summary')
    args = parser.parse_args()

    ignoreFiles = ['default__vector_store.json', 'image__vector_store.json', 
                    'graph_store.json', 'index_store.json', 'docstore.json',
                    'index_store.json'
                ]
    tool = language_tool_python.LanguageTool('es-ES', host='0.0.0.0')
    #tool = language_tool_python.LanguageToolPublicAPI('es-ES', config={'maxSpellingSuggestions': 1})
    listJsons = getDataFiles (args.indir)
    allData = [getFileData (file, tool) for file in listJsons if os.path.basename(file) not in ignoreFiles]
    if args.e:
        print ('Blank summaries will not be removed')
    else:
        print ('Deleting blank summaries.')
        allData = [data for data in allData if data['summary'] != ""]

    df = pd.DataFrame.from_dict(allData)
    df = df.reset_index()
    df = df.rename(columns={"index":"pdf_id"})
    df['pdf_id'] = df.index    
    df.to_parquet (args.outfile, engine='pyarrow')


