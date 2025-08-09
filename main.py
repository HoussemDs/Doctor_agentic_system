import os
from crewai import Crew
from textwrap import dedent
from agents import DoctorsAgents
from tasks import DoctorTasks  # changed from TravelTasks to DoctorTasks
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


class MedicalCrew:
    def __init__(self, patient_data):
        self.patient_data = patient_data

    def run(self):
        # Initialize agents and tasks
        agents = DoctorsAgents()
        tasks = DoctorTasks()

        diagnosis_doctor = agents.Diagnosis_Doctor()
        treatment_doctor = agents.Treatment_Doctor()

        # Create the diagnosis task
        diagnose_task = tasks.diagnose_patient(
            diagnosis_doctor,
            self.patient_data
        )

        # Create the treatment task
        # (In a real flow, you’d pass the actual diagnosis result here,
        # but CrewAI will handle chaining automatically)
        treatment_task = tasks.suggest_treatment(
            treatment_doctor,
            "Diagnosis from previous task"
        )

        # Define the crew
        crew = Crew(
            agents=[diagnosis_doctor, treatment_doctor],
            tasks=[diagnose_task, treatment_task],
            verbose=True,
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    print("## Welcome to the Medical Diagnosis & Treatment Crew")
    print("---------------------------------------------------")
    patient_data = input("Enter patient symptoms and relevant history: ")

    custom_crew = MedicalCrew(patient_data)
    result = custom_crew.run()

    print("\n\n########################")
    print("## Diagnosis & Treatment Result:")
    print("########################\n")
    print(result)
