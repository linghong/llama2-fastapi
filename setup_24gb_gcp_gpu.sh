#!/bin/bash

# Set variables
INSTANCE_NAME=smartchat-24gb
REGION=us-east4
ZONE=us-east4-c
MACHINE_TYPE=g2-standard-4
GPU_TYPE=nvidia-l4
GPU_COUNT=1
BOOT_DISK_NAME=$INSTANCE_NAME
BOOT_DISK_SIZE=200GB
IMAGE_FAMILY=common-cu121-debian-11-py310
IMAGE_PROJECT=deeplearning-platform-release
SERVICE_ACCOUNT=default
ACCELERATOR=nvidia-l4
DEVICE_NAME=instance-1
STATIC_IP_NAME=smartchat-static-ip

# Create a static ip address used for setting up remote ssh in IDE
gcloud compute addresses create $STATIC_IP_NAME --region=$REGION
STATIC_IP_ADDRESS=$(gcloud compute addresses describe $STATIC_IP_NAME --region=$REGION --format="get(address)")

# Create the VM instance
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-device-name=$DEVICE_NAME \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --accelerator type=$GPU_TYPE,count=$GPU_COUNT \
    --maintenance-policy=TERMINATE \
    --no-service-account \
    --no-scopes \
    --tags=http-server,https-server \
    --address=$STATIC_IP_ADDRESS

# Check the exit status of the previous command
if [ $? -eq 0 ]; then
    echo "Instance $INSTANCE_NAME created successfully!"
else
    echo "Error creating instance $INSTANCE_NAME."
fi
