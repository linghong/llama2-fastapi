# SmartChat-fastAPI 
SmartChat-fastAPI serves as the backend for the AI chat platform [SmartChat](https://github.com/linghong/smartchat). It facilitates fine-tuning of AI models and provides operational interfaces for leveraging open-source AI models.

## Features
### Finetuning Request Submission to OpenAI
SmartChat-fastAPI handles [SmartChat](https://github.com/linghong/smartchat) client-side user requests for model fine-tuning. It verifies the incoming data and orchestrates the submission of fine-tuning requests to OpenAI.

### Open Source Language Model REST API Endpoints
This backend hosts selected popular open-source Language Models (LLMs) and exposes REST API endpoints for client-side interactions. Through these endpoints, users can easily access and utilize the hosted LLMs for their applications.

## Setup
This FastAPI application has various dependencies, including some that require the Rust programming language.

### System-Level Dependencies - Rust

  This project requires the Rust programming language for some of its Python dependencies. Install Rust using [rustup](https://rustup.rs/):
  
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs/ | sh
  ```
  
  Follow the on-screen instructions to complete the installation. Once installed, you might need to source your profile to access Rust binaries.

  If you're using bash:
  ```bash
  source ~/.bash_profile
  ```

  If you're using zsh:
  ```
  source ~/.zshenv
  ```
  
  Verify the installation:

  ```bash
  rustc --version
  cargo --version
  ```

### Python Dependencies

  After setting up the system-level dependencies, you can install the Python packages required for this project:

  ```bash
  pip install -r requirements.txt
  ```
## Connecting to Google Cloud Platform (GCP)
1. Install the Google Cloud CLI by following the instructions available at [Google Cloud CLI](https://cloud.google.com/sdk/docs/install). 

2. Navigate to the Google Cloud Console, and create a new project. Once the project is created, set it as your current project by executing the following command in your terminal:
```
gcloud config set project YOUR_PROJECT_ID
```
Replace YOUR_PROJECT_ID with the actual ID of the project you just created.

3. Enable the Compute Engine API for your project. If you plan to use GPU resources, create a GPU instance. Please note that you might need to submit a request to increase your GPU quota if it's insufficient for your needs.

## Running the Application
  start a virtual environment by running the following code
  on Windows:
  ```bash
  .\venv\Scripts\activate
  ```
  on Unix or MacOS:
  ```bash
  source venv/bin/activate
  ```
  You can start the FastAPI application using:
  ```bash
  uvicorn main:app --reload
  ```
