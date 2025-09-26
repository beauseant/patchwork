import pandas as pd
import language_tool_python


tool = language_tool_python.LanguageToolPublicAPI('es-ES')

def correctText ( text,tool ):
	try:
		import ipdb ; ipdb.set_trace()
		textc = tool.correct (text)
	except Exception as E:
		textc = 'NA'
		print (str(E))
		import ipdb ; ipdb.set_trace()

	return textc

data = pd.read_parquet ('/home/sblanco/pliegos/output/2018.parquet')
data['raw_text_c'] = data['raw_text'].apply (lambda row: correctText (row, tool))

data.to_parquet ('/home/sblanco/pliegos/output/2018_c.parquet')
