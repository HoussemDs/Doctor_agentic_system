import os
from crewai import Crew
from agents import DoctorsAgents
from tasks import DoctorTasks
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Set up API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables!")
        print("Please make sure you have a .env file with GROQ_API_KEY=your_key_here")
        return
    
    os.environ["GROQ_API_KEY"] = groq_api_key

    class MedicalCrew:
        def __init__(self, patient_data):
            self.patient_data = patient_data

        def run(self):
            try:
                # Initialize agents and tasks
                agents = DoctorsAgents()
                tasks = DoctorTasks()

                # Create agents
                print("Creating agents...")
                diagnosis_doctor = agents.Diagnosis_Doctor()
                treatment_doctor = agents.Treatment_Doctor()

                # Create tasks
                print("Creating tasks...")
                diagnose_task = tasks.diagnose_patient(
                    diagnosis_doctor,
                    self.patient_data
                )

                treatment_task = tasks.suggest_treatment(
                    treatment_doctor,
                    "Use the diagnosis results from the previous task"
                )

                # Create and run crew
                print("Starting crew execution...")
                crew = Crew(
                    agents=[diagnosis_doctor, treatment_doctor],
                    tasks=[diagnose_task, treatment_task],
                    verbose=True,
                )

                result = crew.kickoff()
                return result
                
            except Exception as e:
                print(f"❌ Error running crew: {e}")
                import traceback
                traceback.print_exc()
                return None

    print("## Heart Diagnosis & Treatment Crew AI with ML Integration")
    print("--------------------------------------------------------")
    print("This system combines medical expertise with machine learning predictions")
    print()
    
    # Test tools first
    print("Testing tools before starting...")
    try:
        from tools import heart_predictor, heart_image_display
        test_result = heart_predictor._run("test")
        print(f"✅ Tools test: {test_result[:50]}...")
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        return
    
    patient_data = input("\nEnter heart-related symptoms and history: ")
    
    print("\nProcessing with AI agents and ML model...")
    print("=" * 50)
    
    medical_crew = MedicalCrew(patient_data)
    output = medical_crew.run()
    
    print("\n" + "=" * 50)
    if output:
        print("########## FINAL RESULTS ##########")
        print(output)
    else:
        print("❌ No results generated. Check the errors above.")

if __name__ == "__main__":
    main()