from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import warnings
warnings.filterwarnings('ignore')

# Initialize the model 
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
            Decompose it into a list of smaller, actionable sub-tasks, each on a new line.
            
            User Query: {query}
            
            Sub-tasks:
            """,
        )
        self.task_decomposition_chain = LLMChain(
            llm=self.llm, prompt=self.task_decomposition_template
        )

        self.tools = {
            "research": self.mock_research,
            "book": self.mock_book, 
            "create": self.mock_create_itinerary 
        }

    def decompose_task(self, query):
        response = self.task_decomposition_chain.run(query=query)
        sub_tasks = response.strip().split("\n")
        return sub_tasks

    def execute_task(self, task):
        tool_name = task.split(" ", 1)[0].lower() 
        tool_input = task.split(" ", 1)[1] if len(task.split(" ", 1)) > 1 else ""
        if tool_name in self.tools:
            return self.tools[tool_name](tool_input)
        else:
            return f"Error: No tool available for '{task}'"

    def mock_research(self, input):
        return f"Researched: {input}. Found some good options!"

    def mock_book(self, input):
        return f"Booked: {input}"

    def mock_create_itinerary(self, input):
        return f"Created a basic itinerary for {input}"

# Example usage:
plan_agent = PlanAgent(llm)
user_query = "Plan a honeymoon trip to Rome for 7 days."
sub_tasks = plan_agent.decompose_task(user_query)

# Iterative Refinement Loop 
for i in range(3):  
    for j, task in enumerate(sub_tasks):
        print(f"**Sub-task {j+1}:** {task}")
        result = plan_agent.execute_task(task)
        print(f"Result: {result}")

        # Feedback mechanism 
        feedback = {
            "status": "success" if "Error" not in result else "failed",
            "message": result,
            "task_index": j, 
        }

        # Task Management Logic
        if feedback["status"] == "failed":
            if "No tool available" in feedback["message"]:
                print("Attempting to rephrase the task...")
                print(f"Task before rephrasing: {task}")
                try:
                    rephrased_task = llm(f"Rephrase this task to be easier to understand: {str(task)}").text
                    print(f"Rephrased task: {rephrased_task}")
                    sub_tasks.insert(j + 1, rephrased_task) 
                except Exception as e:
                    print(f"Error during rephrasing: {e}") 
                    # Add more detailed error handling or logging here
                break 