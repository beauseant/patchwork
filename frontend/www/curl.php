curl -X 'POST' \
  'http://kumo01.tsc.uc3m.es:112/pdf/extract_text/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@a6e6aa4f77315da1f217cc45a4ba0724.pdf;type=application/pdf'
