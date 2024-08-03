import os
import requests
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI


os.environ["OPENAI_API_BASE"] = 'http://localhost:11434/v1'
os.environ["OPENAI_MODEL_NAME"] = 'phi3:3.8b'
os.environ["OPENAI_API_KEY"] = 'sk-111111111111111111111111111111111111111111111111'

def fetch_wikipedia_content():
    """ Fetch the main content of the Bangladesh Wikipedia page """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": "Bangladesh",
        "format": "json",
        "prop": "text"
    }
    response = requests.get(url, params=params)
    html_content = response.json()['parse']['text']['*']
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup.find_all(['sup', 'table', 'script']):
        tag.decompose()
    
    return soup.get_text()


wikipedia_content = fetch_wikipedia_content()
print(wikipedia_content[:500]) 


researcher = Agent(
  role='Research Analyst',
  goal='Extract relevant information from the Wikipedia page on Bangladesh.',
  backstory="""You are an experienced research analyst proficient in extracting and summarizing information from large documents.""",
  verbose=True,
  allow_delegation=False,
  llm=ChatOpenAI(model_name="phi3:3.8b", temperature=0.7),
)

writer = Agent(
  role='Q&A System Developer',
  goal='Develop a system that can answer questions about Bangladesh using the extracted information.',
  backstory="""You are a developer skilled in creating interactive Q&A systems using AI models.""",
  verbose=True,
  allow_delegation=True,
  llm=ChatOpenAI(model_name="phi3:3.8b", temperature=0.7),
)


task1 = Task(
  description="Extract relevant content from the Wikipedia page on Bangladesh.",
  expected_output="Extracted content in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the extracted content, develop a Q&A system that can answer questions about Bangladesh.
  The system should be able to handle queries and return accurate answers based on the content provided.""",
  expected_output="Functioning Q&A system capable of answering questions about Bangladesh",
  agent=writer
)


crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2,  
  process=Process.sequential
)


result = crew.kickoff()

print("######################")
print(result)


def answer_question(question):
    context = f"Based on the Wikipedia page on Bangladesh, here is the relevant content:\n{wikipedia_content}"
    complete_question = f"{context}\nQuestion: {question}"
    response = writer.llm.invoke(complete_question)
    return response


while True:
  inp=input("Ask Question or STOP:")
  if inp!="STOP":
    question = inp
    answer = answer_question(question)
    print(f"Q: {question}\nA: {answer}")
  else:
     break