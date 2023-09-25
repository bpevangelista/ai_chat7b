# install kubernets cli
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
	&& sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
	&& rm kubectl

# connect kubectl to azure
az aks get-credentials --resource-group inference-sandbox0 --name test1

# list nodes (should see agent and user)
kubectl get nodes

# create service
kubectl apply -f test1-service.yaml
