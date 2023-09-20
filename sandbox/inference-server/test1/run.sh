#docker build . -t image.name.here
#docker images
#docker rmi blah
#docker run --name test1 azure.inference.test1

gunicorn app:app -b 0.0.0.0:8080 --log-file - --access-logfile - --workers 4 --keep-alive 0
