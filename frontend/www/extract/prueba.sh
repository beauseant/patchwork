curl -X 'POST' \
  'http://kumo01:9084/objective/extract/fromFile/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/tmp/prueba.txt;type=text/plain'

curl -X 'POST' \
   'http://kumo01:9084/objective/extract/fromFile/' \
   -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
   -F 'file=@/tmp/prueba.txt;type=text/plain'
