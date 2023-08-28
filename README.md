# Llama2-fastAPI Project

This project is a FastAPI application with various dependencies, including some that require the Rust programming language.

## Setup

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

## Running the Application
  You can start the FastAPI application using:

  ```bash
  uvicorn main:app --reload
  ```