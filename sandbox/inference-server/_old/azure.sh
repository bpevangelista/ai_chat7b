# install kubernets cli
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
	&& sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
	&& rm kubectl

# create resource group
az login
az group create --name inference-sandbox0 --location westus2

# create container registry
az acr create --resource-group inference-sandbox0 --name evangelista --sku Basic --location westus2
az acr update -n evangelista --admin-enabled true

# push repository to container
az acr login --name evangelista
docker push evangelista.azurecr.io/test1

# get kubernets credentials for lens desktop
az aks get-credentials --resource-group inference-sandbox0 --name test1

