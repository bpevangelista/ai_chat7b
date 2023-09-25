
docker build . -t evangelista.azurecr.io/test1 --platform linux/arm64
docker images
#docker rmi blah

#az login
#az group create --name inference-sandbox0 --location westus2
#az acr create --resource-group inference-sandbox0 --name evangelista --sku Basic --location westus2 --admin-enabled true
#az acr update -n evangelista --admin-enabled true

#az acr login --name evangelista
#docker push evangelista.azurecr.io/test1

docker run --name test1 azure.inference.test1
