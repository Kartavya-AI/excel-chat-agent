from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
load_dotenv()

print(os.getenv("GEMINI_API_KEY"))
# Initialize Gemini model
llm = LLM(model="gemini/gemini-2.0-flash")

@CrewBase
class excelcrew():
    """excelcrew crew"""

    agents_config = 'config/agent.yaml'
    tasks_config = 'config/tasks.yaml'

    # === Agents ===

    @agent
    def data_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyzer'],
            llm=llm
        )

    @agent
    def data_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config['data_reporter'],
            llm=llm
        )

    # === Tasks ===
    @task
    def analyze_data_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_data_task"],
            agents=[self.data_analyzer],
        )

    @task
    def report_data_task(self) -> Task:
        return Task(
            config=self.tasks_config["report_data_task"],
            agents=[self.data_reporter],
        )
    # === Crew ===
    @crew
    def crew(self) -> Crew:
        """Creates the excelcrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose= True,
        )
