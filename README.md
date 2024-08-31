# GUC Chat

## Description

This chat application leverages the capabilities of Gemini, a large language model, and prompt engineering techniques to create a dynamic and informative conversational experience. The application distinguishes itself by its exceptional flexibility and adaptability, facilitated by a Retrieval Augmented Generation (RAG) system. This enables the system to quickly incorporate, modify, or remove information from its knowledge base, ensuring that the chatbot remains up-to-date and provides accurate, relevant responses.

The chatbot can then be asked about this data, offering quick and accurate responses based on the most recent information available. This flexibility makes it ideal for use cases where the underlying data needs to be frequently updated or customized, such as customer support, knowledge bases, or interactive learning platforms.

## Demo


https://github.com/user-attachments/assets/2f0b3a8e-b215-4aad-9c9c-ca999b4a738d



## Installation

### Setup

# 0. Ensure Python 3.x is installed. If not, install it:
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip
```

# 1. Clone the repository:
```bash
git clone https://github.com/NourAlPha/guc-chat
cd guc-chat
```

# 2. Install `venv` if you don't have it:
```bash
python3 -m pip install --user virtualenv
```

# 3. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

# 4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

# 5. Install additional system packages from `packages.txt`:
```bash
xargs sudo apt-get install -y < packages.txt
```

# 6. Get a Gemini API key and put it in environment variables:
  1. Get API key from: https://aistudio.google.com/app/apikey
  2. Do the following script:
     ```bash
     nano .bashrc
     (in the last line add)
     export GOOGLE_API_KEY="YourActualAPIKey" (without quotes)
     (save and exit)
     source .bashrc
     ```

## Run the Application
```bash
streamlit run Home.py
```
