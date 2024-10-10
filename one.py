from langchain_groq import ChatGroq
from langchain import PromptTemplate, LLMChain

import warnings
warnings.filterwarnings('ignore')

# Initialize the model with parameters
llm = ChatGroq(
    model_name="llama-3.2-90b-text-preview",  
    temperature=0,
    groq_api_key="API-Key"
)

class PlanAgent:
    def __init__(self, llm):
        self.llm = llm
        self.task_decomposition_template = PromptTemplate(
            input_variables=["query"],
            template="""You are a helpful AI agent. You are given a user query. 
            Decompose it into a list of smaller, actionable sub-tasks.
            
            User Query: {query}
            
            Sub-tasks:
            - Research potential destinations for a honeymoon.
            - Book flights and accommodation.
            - Create a rough itinerary of activities.
            
            """,
        )
        self.task_decomposition_chain = LLMChain(
            llm=self.llm, prompt=self.task_decomposition_template
        )

    def decompose_task(self, query):
        response = self.task_decomposition_chain.run(query=query)
        # Process the response to extract the list of sub-tasks
        sub_tasks = response.strip().split("\n")
        return sub_tasks

# Example usage:
llm = llm
plan_agent = PlanAgent(llm)
user_query = "Plan a trip to Paris for 5 days with a budget of $2000."
sub_tasks = plan_agent.decompose_task(user_query)
print(sub_tasks)
