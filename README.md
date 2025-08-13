# Heart Diagnosis & Treatment AI System

This project is an **AI-powered medical diagnosis and treatment system** designed to assist healthcare professionals. It combines human-like AI agents, machine learning models, and visual medical references to provide a comprehensive and accurate analysis of heart conditions.

### 🎯 High-Level Overview

The system's core functionality can be broken down as follows:

* **Input**: The system takes patient symptoms and medical history as text.
* **Process**: A "Diagnosis Doctor" AI agent uses a machine learning model to analyze the data and a "Treatment Doctor" AI agent creates a personalized treatment plan. The system also displays relevant medical images for context.
* **Output**: A precise diagnosis, a detailed treatment plan, and visual aids for reference.

**Key Benefits:**

* **For Medical Professionals**: Provides a quick second opinion, consistent treatment recommendations, and visual aids for patient education.
* **Technical Innovation**: Integrates AI reasoning with ML predictions and uses a multi-agent approach to mimic a medical team.
* **Business Value**: Reduces diagnosis time, improves accuracy through a dual-validation process, and offers a scalable healthcare support solution.

---

### 🔧 File-by-File Breakdown

This project is structured for clarity and modularity. Here's a breakdown of the key files and their roles:

* `main.py`: The system's entry point. It orchestrates the entire workflow, from loading environment variables to coordinating the AI agents and tasks, and finally, presenting the results.
* `agents.py`: Defines the two primary AI agents: the **Diagnosis Doctor** and the **Treatment Doctor**. Each agent has a specific role, a professional persona (e.g., "cardiologist with 20 years of experience"), and a set of goals.
* `tasks.py`: Contains the medical procedures the agents must perform. It defines the `diagnose_patient` and `suggest_treatment` tasks, detailing the steps and expected outputs for each.
* `tools.py`: The bridge between the AI agents and the machine learning models. It defines tools like the **`HeartPredictionTool`**, which loads and runs a pre-trained Random Forest model, and the **`HeartConditionImageTool`**, which displays relevant medical imagery.
* `requirements.txt`: Lists all the necessary Python dependencies for the project.
* `Random_Forest_Heart_Model.joblib` & `model_columns.joblib`: These are the saved files for the trained machine learning model and its feature columns, respectively.
* `images/`: A directory containing medical images used to visually represent heart conditions.

---

### 🎭 AI Agents - The Digital Doctors

The core of this system is the multi-agent architecture, defined in **`agents.py`**.

* **Diagnosis Doctor**: An agent with a cardiologist persona specializing in diagnosis. Its goal is to identify heart problems by using its medical expertise and a specialized ML prediction tool. It outputs a clear diagnosis, such as "Sick with [condition] affecting [heart part]."
* **Treatment Doctor**: A senior cardiologist focused on creating comprehensive treatment plans. Its goal is to use the diagnosis from the first agent to suggest medications, lifestyle changes, and follow-up plans.

Both agents are built using the **CrewAI framework** and leverage **Groq's LLaMA model** for their reasoning capabilities.

---

### 📋 Tasks - The Medical Procedures

The **`tasks.py`** file defines the step-by-step procedures the AI agents follow to complete their work.

* **`diagnose_patient`**: This task instructs the Diagnosis Doctor to analyze symptoms, use the ML prediction tool for data-driven insights, combine those insights with its own medical knowledge, and, if a condition is found, use the image tool to find relevant visuals before providing a final diagnosis.
* **`suggest_treatment`**: This task guides the Treatment Doctor to create a detailed, personalized treatment plan based on the diagnosis received from the first task.

---

### 🛠️ ML & Tool Integration

The **`tools.py`** file is crucial for connecting the AI agents to external data and models.

* **`HeartPredictionTool`**: This tool loads a pre-trained **Random Forest classifier** trained on over 40 medical features. When an agent calls this tool, it processes the patient data and returns a prediction for one of 13 different heart conditions.
* **`HeartConditionImageTool`**: This tool maps a diagnosed condition to a corresponding image from the `images/` directory, helping to visually support the diagnosis and aid patient understanding.

---

### 🚀 Getting Started

To run this project, you'll need to:

1.  **Clone the repository**.
2.  **Install dependencies** using `pip install -r requirements.txt`.
3.  **Set up your API key**: Get a Groq API key and add it to a `.env` file at the root of the project as `GROQ_API_KEY="your_api_key_here"`.
4.  **Run the main script** with `python main.py`.

The system will then run the pre-flight checks, create the medical crew, execute the diagnosis and treatment pipeline, and display the results.
