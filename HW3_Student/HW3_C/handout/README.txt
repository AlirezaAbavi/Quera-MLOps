HW3_C Credentials for alireza_abouei
================================

KUBECONFIG:  $(pwd)/kubeconfig.yaml
NAMESPACE:   qbc12-hw03-c-alireza_abouei
NODEPORT:    30097
REGISTRY:    185.50.38.163:35000
SERVICE_URL: http://185.50.38.163:30097

Steps:
1. export KUBECONFIG=$(pwd)/kubeconfig.yaml
2. kubectl get nodes
3. docker pull 185.50.38.163:35000/qbc12-hw03-base:v1
4. docker build -t 185.50.38.163:35000/qbc12-embedder-alireza_abouei:v1 .
5. docker push 185.50.38.163:35000/qbc12-embedder-alireza_abouei:v1
6. kubectl apply -f k8s/
7. curl http://185.50.38.163:30097/healthz/ready
