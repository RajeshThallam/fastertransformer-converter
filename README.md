# Serving Large Language Models on Google Kubernetes Engine (GKE) using NVIDIA Triton Inference Server with FasterTransformer

This repository compiles prescriptive guidance and code sample to serve Large Language Models (LLM) such as [Unified Language Learner (UL2)](https://ai.googleblog.com/2022/10/ul2-20b-open-source-unified-language.html) on a Google Kubernetes Engine (GKE) cluster with GPUs running NVIDIA Triton Inference Server with FasterTransformer backend.

- [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) is an open-source inference serving solution from NVIDIA to simplify and standardize the inference serving process supporting multiple frameworks and optimizations for both CPUs and GPUs.
- [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer/) library implements an accelerated engine for the inference of transformer-based models, spanning multiple GPUs and nodes in a distributed manner.

The solution provides a Terraform standardized template to [deploy Triton inference server on GKE](https://github.com/jarokaz/triton-on-gke-sandbox) and integrate with other Google Cloud Managed Services.


## High Level Flow

The solution demonstrates deploying the UL2 (20B parameter) model on a GKE cluster with GPUs. Assuming, JAX based checkpoints of a [pre-trained](https://github.com/google-research/google-research/tree/master/ul2#checkpoints) or fine-tuned UL2 model are available, the workflow has the following steps:

1. Set up the environment running [Triton server on GKE cluster](https://github.com/jarokaz/triton-on-gke-sandbox).
2. Convert JAX checkpoint to FasterTransformer checkpoint
3. Serve the resultant model on GPUs using NVIDIA Triton Inference server with FasterTransformer backend
4. Run evaluation script with test instances to compute model eval metrics

![flow](/images/flow.png)

## Checkpoints

You have following ways to access JAX based checkpoints for UL2:

- You can find pre-trained UL2 checkpoints [here](https://github.com/google-research/google-research/tree/master/ul2#checkpoints).
- We fined-tuned UL2 model with [XSum dataset](https://www.tensorflow.org/datasets/catalog/xsum) and made checkpoints available on Google Cloud Storage bucket at `gs://se-checkpoints/ul2-xsum/`.

---

**NOTE:** You can refer to the following [solution accelerator](https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai) to fine-tune UL2 model with custom datasets using [T5X framework](https://github.com/google-research/t5x) that creates JAX based checkpoints. 

---

## Getting started

### 1. Clone the GitHub repo

``` bash
cd ~
git clone https://github.com/RajeshThallam/fastertransformer-converter
cd ~/fastertransformer-converter
```

### 2. Environment setup

Follow the environment setup guide [here](https://github.com/jarokaz/triton-on-gke-sandbox) to create a GKE cluster running NVIDIA Triton on GPU node pool using Terraform standardized template. The setup performs the following steps:

- [ ] Enable APIs
- [ ] Run Terraform to provision the required resources
- [ ] Deploy Ingress Gateway
- [ ] Deploy NVIDIA GPU drivers
- [ ] Configure and deploy Triton Inference Server
- [ ] Run health check to validate the Triton deployment

As part of the environment setup, you configure following environment variables:

``` bash
export PROJECT_ID=my-project-id
export REGION=us-central1
export ZONE=us-central1-a
export NETWORK_NAME=my-gke-network
export SUBNET_NAME=my-gke-subnet
export GCS_BUCKET_NAME=my-triton-repository
export GKE_CLUSTER_NAME=my-ft-gke
export TRITON_SA_NAME=triton-sa
export TRITON_NAMESPACE=triton
```

<div class="alert alert-block alert-warning"> 
    <p>
        <strong>⚠️ NOTE: </strong>
Before running through the setup, ensure the GPU node pool in the GKE cluster is configured with minimum 1 NVIDIA A100 GPU by setting the following variables. This is prerequisite for deploying large model, such as UL2, which is deployed with `bfloat16` (BF16) activation in this tutorial.
    </p>
</div>

``` bash
export MACHINE_TYPE=a2-highgpu-1g
export ACCELERATOR_TYPE=nvidia-tesla-a100
export ACCELERATOR_COUNT=1
```

After provisioning the GKE cluster, configure access to the cluster:

``` bash
gcloud container clusters get-credentials ${GKE_CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE} 

kubectl create clusterrolebinding cluster-admin-binding --clusterrole cluster-admin --user "$(gcloud config get-value account)"
```

### 2. Convert JAX checkpoint to FasterTransformer checkpoint

The checkpoint format conversion from JAX to NVIDIA FasterTransformer is run on GKE cluster as a kubernetes Job on the GPU node pool. The source code for conversion script is located [here](./converter).

- Create Docker repository in Google Artifact Registry to manage images

``` bash
# Configure parameters 
export DOCKER_ARTIFACT_REPO=llm-inference     # <-- Change to your repo name

# Enable API
gcloud services enable artifactregistry.googleapis.com

# Create repository
gcloud artifacts repositories create ${DOCKER_ARTIFACT_REPO} \
    --repository-format=docker \
    --location={REGION} \
    --description="Triton Docker repository"

# Authenticate to repository
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
```

- Build container image

``` bash
# Configure container image name
export JAX_TO_FT_IMAGE_NAME="jax-to-fastertransformer"
export JAX_TO_FT_IMAGE_URI=${REGION}"-docker.pkg.dev/"${PROJECT_ID}"/"${DOCKER_ARTIFACT_REPO}"/"${JAX_TO_FT_IMAGE_NAME}

# Change directory
cd ~/fastertransformer-converter

# Run Cloud Build job to build the container image
export FILE_LOCATION="./converter"
gcloud builds submit \
  --region ${REGION} \
  --config converter/cloudbuild.yaml \
  --substitutions _IMAGE_URI=${JAX_TO_FT_IMAGE_URI},_FILE_LOCATION=${FILE_LOCATION} \
  --timeout "2h" \
  --machine-type=e2-highcpu-32 \
  --quiet
```

- Copy UL2 checkpoints to your Google Cloud Storage bucket

``` bash
gcloud storage cp -r gs://se-checkpoints/ul2-xsum/ gs://${GCS_BUCKET_NAME}/models/
```

- Run conversion job start with configuring job parameters

``` bash
cd ~/fastertransformer-converter/env-setup/kustomize
  
cat << EOF > ~/fastertransformer-converter/converter/configs.env
ksa=triton-sa
converter_image_uri=$JAX_TO_FT_IMAGE_URI
accelerator_count=1
model_name=ul2
gcs_jax_ckpt=gs://${GCS_BUCKET_NAME}/models/ul2-xsum/
gcs_ft_ckpt=gs://${GCS_BUCKET_NAME}/triton_model_repository/ul2-ft/
EOF
```

- Deploy the configuration

``` bash
kubectl kustomize ./
```

- Run the job

``` bash
kubectl apply -k ./
```

- Monitor the job

``` bash
kubectl logs
```

### 3. Serve FasterTransformer checkpoint with NVIDIA Triton

- Pull NVIDIA NeMo Inference container with NVIDIA Triton and FasterTransformer
  - Sign in to [NGC](https://ngc.nvidia.com/signin) and select organization as `ea-participants`
  - Get [API key from NGC](https://ngc.nvidia.com/setup/api-key ) to authorize docker client to access NGC Private Registry
  - Authorize docker

``` bash
docker login nvcr.io

Username: $oauthtoken
Password: Your key
```
-  
  - Push the container to your Docker repository in Cloud Artifact Registry 

``` bash
export TRITON_FT_IMAGE_URI=${REGION}"-docker.pkg.dev/"${PROJECT_ID}"/"${DOCKER_ARTIFACT_REPO}"/bignlp-inference:22.08-py3"

docker pull nvcr.io/ea-bignlp/bignlp-inference:22.08-py3

docker tag nvcr.io/ea-bignlp/bignlp-inference:22.08-py3 ${TRITON_FT_IMAGE_URI}

docker push ${TRITON_FT_IMAGE_URI}
```

- Configure NVIDIA Triton Deployment parameters

``` bash
cd ~/triton-on-gke-sandbox/env-setup/kustomize

cat << EOF > ~/triton-on-gke-sandbox/env-setup/kustomize/configs.env
model_repository=gs://${GCS_BUCKET_NAME}/triton_model_repository/ul2-ft/
ksa=${TRITON_SA_NAME}
EOF
```

- Update NVIDIA Triton container image

``` bash
kustomize edit set image "nvcr.io/nvidia/tritonserver:22.01-py3="${TRITON_FT_IMAGE_URI}
```

- Deploy the configuration

``` bash
kubectl kustomize ./
```

- Deploy to the cluster

``` bash
kubectl apply -k ./
```

- Run health check

To validate that NVIDIA Triton Inference Server has been deployed successfully try to access the server's health check API. You will access the server through Istio Ingress Gateway. Start by getting the external IP address of the  `istio-ingressgateway` service.

```bash
kubectl get services -n $TRITON_NAMESPACE

# Invoke the health check API
ISTIO_GATEWAY_IP_ADDRESS=$(kubectl get services -n $TRITON_NAMESPACE \
   -o=jsonpath='{.items[?(@.metadata.name=="istio-ingressgateway")].status.loadBalancer.ingress[0].ip}')

curl -v ${ISTIO_GATEWAY_IP_ADDRESS}/v2/health/ready
```

If the returned status is `200OK` the server is up and accessible through the gateway.

### 4. Run evaluation 



## Repository Structure

```
.
├── converter
├── evaluator
└── README.md
```

- `/converter`: Source for converting JAX checkpoints to FasterTransformer checkpoints
- `/evaluator`: Source for running model evaluation on validation dataset with model hosted on NVIDIA Triton

## Getting help
If you have any questions or if you found any problems with this repository, please report through GitHub issues.

## Citations
- XSum
    ```
    @InProceedings{xsum-emnlp,
    author =      "Shashi Narayan and Shay B. Cohen and Mirella Lapata",
    title =       "Don't Give Me the Details, Just the Summary! {T}opic-Aware Convolutional Neural Networks for Extreme Summarization",
    booktitle =   "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing ",
    year =        "2018",
    address =     "Brussels, Belgium",
    }
    ```
- UL2
    ```
    @article{tay2022unifying,
    title={Unifying Language Learning Paradigms},
    author={Yi Tay*, Mostafa Dehghani*, Vinh Q. Tran, Xavier Garcia, Dara Bahr, Tal Schuster, Huaixiu Steven Zheng, Neil Houlsby, and Donald Metzler},
    year={2022}
    }
    ```