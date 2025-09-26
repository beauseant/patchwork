import pathlib
from src.pdf_parser import PDFParser
from src.utils import clean_string

import argparse

def getPages (pages):
    return [ (page['element_content']) for page in  pages ]

def correctText (text):
    lang_tool = language_tool_python.LanguageTool('es-ES', remote_server='http://0.0.0.0:8081') # type: ignore
    try:
        textc = lang_tool.correct (text)
    except:
        textc = 'NA'
    return textc

def getRawText (data ):
    output = {}
    output['metadata'] = data['metadata']
    output['raw_text'] = " ".join (str(element[0]) for element in [ getPages (page['content']) for page in data['pages']])
    #textc = clean_string (output['raw_text']).encode().decode('UTF-8', errors='ignore').split()
    '''
    textc = clean_string (output['raw_text']).split()

    num = 1000
    chunks = [textc[i:i + num] for i in range(0, len(textc), num)]
    #lang_tool = language_tool_python.LanguageTool('en-US', remote_server='http://0.0.0.0:8081')
    #import ipdb ; ipdb.set_trace()
    #textc = lang_tool.correct (text)
    chunksc = [ correctText (' '.join (chunk)) for chunk in chunks]
    #import ipdb ; ipdb.set_trace()
    output['raw_textc'] =  ' '.join(str(chunk) for chunk in chunksc)'''
    return (output['raw_text'])

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file', required=True)
    parser.add_argument('--path_save', type=str, help='Path to save the extracted content', required=True)
    parser.add_argument('--output', help='Output raw text to tty and file, default false', action=argparse.BooleanOptionalAction)
    parser.set_defaults(output=False)
    parser.add_argument('--summary', help='Create summary using llm, default false', action=argparse.BooleanOptionalAction)
    parser.set_defaults(summary=False)


    args = parser.parse_args()

    pdf_file = pathlib.Path(args.pdf_path)
    path_save = pathlib.Path(args.path_save) / pdf_file.stem
    output = args.output

    # Create a directory for the extracts (one directory per PDF file)
    path_save.mkdir(parents=True, exist_ok=True)
    path_save.joinpath("images").mkdir(parents=True, exist_ok=True)
    path_save.joinpath("tables").mkdir(parents=True, exist_ok=True)

    # Create a PDFParser and parse the PDF file
    pdf_parser = PDFParser(
        extract_header_footer=False,
        generate_img_desc=False,
        generate_table_desc=False,
    )

    document = pdf_parser.parse(pdf_path=pdf_file, path_save=path_save)

    if (args.output):
        print(getRawText ( document ))

    return


if __name__ == "__main__":
    main()
