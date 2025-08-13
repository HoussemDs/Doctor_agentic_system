from crewai import Task
from textwrap import dedent

class DoctorTasks:
    def __tip_section(self):
        return "Focus only on heart-related symptoms and conditions."

    def diagnose_patient(self, agent, patient_data):
        return Task(
            description=dedent(
                f"""
                **Task**: Diagnose Heart Condition
                **Description**: Analyze the patient's heart-related symptoms and identify if there is a problem.
                    
                    IMPORTANT: Use the "Heart Disease Predictor" tool to get ML-based predictions 
                    and combine this with your medical expertise.
                    
                    Steps:
                    1. First, use the Heart Disease Predictor tool with the patient data
                    2. Analyze the ML prediction results
                    3. Combine ML insights with your medical knowledge
                    4. If a specific condition is predicted, use the Heart Condition Image Display tool to show relevant imagery
                    5. Provide final diagnosis
                    
                    Output format: Either "Healthy heart" or "Sick with [specific heart condition and affected part]".
                
                **Patient Data**: {patient_data}
                
                **Available Tools**: 
                - Heart Disease Predictor: Use this to get ML-based diagnosis
                - Heart Condition Image Display: Use this to show relevant medical imagery
                
                **Note**: {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output='Either "Healthy heart" or "Sick with [heart condition] affecting [heart part]", including ML prediction results and medical imagery when applicable.'
        )

    def suggest_treatment(self, agent, diagnosis):
        return Task(
            description=dedent(
                f"""
                **Task**: Suggest Heart Treatment Plan
                **Description**: Based on the diagnosis from the previous task, suggest a comprehensive treatment plan.
                    Focus on the specific heart condition identified and the affected heart parts.
                    
                    Include:
                    - Specific medications for the diagnosed condition
                    - Lifestyle changes tailored to the heart condition
                    - Treatment duration and follow-up schedule
                    - Emergency signs to watch for
                    
                **Diagnosis from previous task**: {diagnosis}
                
                **Note**: {self.__tip_section()}
                """
            ),
            agent=agent,
            expected_output='Comprehensive treatment plan with specific medications, lifestyle recommendations, and monitoring schedule tailored for the diagnosed heart condition.'
        )