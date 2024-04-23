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

def get_breed_info(
    breed_id: int = 1,
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




agent = reasoning_engines.LangchainAgent(
    model=model,  # Required.
    tools=[get_breed_info,search_tool],  # Optional. See "Define Python functions"
    model_kwargs=model_kwargs,  # Optional. See "Configure Model"
)



prompt = PromptTemplate(input_variables = ['input'],                            
          template = '''

           You have a great sense of humor. Use the information collected from the tools to present your response in a witty tone.  

            You have been given access to a dog breed info tool. Please use this tool to gather information about the dog breed if the the query: ## {input} ##, is about dogs.  

            You have been given access to a search tool. Please use this tool to gather any additoinal information not available dog breed info tool to answer the query.  

            You will maintain a log of your decisions on when to use which tool. You will print this decision log along with the ouput
                   
           '''                                                                      
        )  


query="What dog breed has the longest life span?"
print(prompt.format(input=query))
response = agent.query(input=prompt.format(input=query))
print(response.get('output'))



DISPLAY_NAME = "Pet Info Langchain Application"

remote_app = reasoning_engines.ReasoningEngine.create(
    reasoning_engines.LangchainAgent(
        model=model,
        tools=[get_breed_info,search_tool],
        model_kwargs=model_kwargs,
    ),
    requirements=[
        "google-cloud-aiplatform",
        "langchain",
        "langchain-core",
        "langchain-google-vertexai",
        "requests==2.*",
        "langchain_community"
    ],
    display_name=DISPLAY_NAME,
)
remote_app

remote_app.operation_schemas()



