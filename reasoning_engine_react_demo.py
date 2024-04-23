import vertexai
from vertexai.preview import reasoning_engines
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import requests
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
import pprint
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    grounding,
    Tool,
)

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents.format_scratchpad import (
    format_to_openai_function_messages
)
from langchain_core.tools import render_text_description
from langchain.tools.base import StructuredTool

PROJECT_ID="vipin-genai-bb"
LOCATION="us-central1"
BUCKET="gs://llm-staging"

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET,
)

model = "gemini-1.0-pro"

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

tool_g = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
model_kwargs = {
    "temperature": 0.28,
    "max_output_tokens": 5000,
    "top_p": 0.95,
    "top_k": 40,
    "safety_settings": safety_settings,

}

def search_tool():
    
    """Retrieves information using internet search. 
    
    

    Args: Query to be used for search
        

    Returns:
        text: response from search
    """


    search_api_key="ENTER-API-KEY"
    os.environ["SERPER_API_KEY"] = search_api_key


    google_search = GoogleSerperAPIWrapper()

    response = google_search.run("query")
    return response.json()

def get_pet_info(

):
    
    """Retrieves a list of dog breeds and their information. 
    
    It contains an array of json objects. Each json object in the has information about dog breed. The dog breed information will contain details around 
    weight, hight, what is it bred for, breed group, life span, temperament and breed name.

    Uses the thedogapi API (https://api.thedogapi.com/) to obtain information across dog breeds

    Args: This function does not take any argument
        

    Returns:
        dict: A list of json objects containing information across dog breeds.
             Example: [{"weight":{"imperial":"6 - 13","metric":"3 - 6"},"height":{"imperial":"9 - 11.5","metric":"23 - 29"},"id":1,"name":"Affenpinscher","bred_for":"Small rodent hunting, lapdog","breed_group":"Toy","life_span":"10 - 12 years","temperament":"Stubborn, Curious, Playful, Adventurous, Active, Fun-loving","origin":"Germany, France","reference_image_id":"BJa4kxc4X"},{"weight":{"imperial":"50 - 60","metric":"23 - 27"},"height":{"imperial":"25 - 27","metric":"64 - 69"},"id":2,"name":"Afghan Hound","country_code":"AG","bred_for":"Coursing and hunting","breed_group":"Hound","life_span":"10 - 13 years","temperament":"Aloof, Clownish, Dignified, Independent, Happy","origin":"Afghanistan, Iran, Pakistan","reference_image_id":"hMyT4CDXR"}]
    """
    
    

    API_KEY="ENTER-API-KEY"

    response = requests.get(
        f"https://api.thedogapi.com/v1/breeds/",
        params={"api-key": API_KEY},
    )
    return response.json()

prompt = hub.pull("hwchase17/react")



tools = [
    StructuredTool.from_function(get_pet_info),
    StructuredTool.from_function(search_tool),
]
agent = reasoning_engines.LangchainAgent(
    model=model,  # Required.
    tools=tools,
    model_kwargs=model_kwargs,  # Optional. See "Configure Model"
    prompt={
        "input": lambda x: x["input"],
        "tool_names": lambda x: ", ".join([t.name for t in tools]),
        "tools": lambda x: render_text_description(tools),
        "agent_scratchpad": (
            lambda x: format_to_openai_function_messages(x["intermediate_steps"])
        ),
    } | prompt,
)
query="I want a pet dog that is moderate on energy, loves to cuddle and gaurd if required. Which breed would you recommend? I live in an apartment.I have about an hour to exercise my do. I do not have children. I am a first time dog owner. I have a parrot as a pet."

#print(prompt.format(input=input1))
response = agent.query(input=query)

pprint.pp(response)

print()

reasoning_engine = vertexai.preview.reasoning_engines.ReasoningEngine('projects/Project_id/locations/us-central1/reasoningEngines/Resource_ID')

input2="Which is the tallest breed?"
response = reasoning_engine.query(input=prompt.format(input=input2))

pprint.pp(response)



