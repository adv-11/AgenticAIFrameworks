from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    
)

model = "gpt-4o-mini"

system_prompt = "You are a chatbot."


#A simple agent's context window

chat_completion = client.chat.completions.create(
    model=model,
    
    messages=[
        # system prompt: always included in the context window 
        {"role": "system", "content": system_prompt}, 
        # chat history (evolves over time)
        {"role": "user", "content": "What is my name?"}, 
    ]
)
print(chat_completion.choices[0].message.content)


agent_memory = {"human": "Name: Bob"}
system_prompt = "You are a chatbot. " \
+ "You have a section of your context called [MEMORY] " \
+ "that contains information relevant to your conversation"

import json


chat_completion = client.chat.completions.create(
    model=model,
    messages=[
        # system prompt 
        {"role": "system", "content": system_prompt + "[MEMORY]\n" + json.dumps(agent_memory)},
        # chat history 
        {"role": "user", "content": "What is my name?"},
    ],
)

print(chat_completion.choices[0].message.content)


#Defining a memory editing tool
#Adding memory to the context 
agent_memory = {"human": "", "agent": ""}

def core_memory_save(section: str, memory: str): 
    agent_memory[section] += '\n' 
    agent_memory[section] += memory 




# tool description 
core_memory_save_description = "Save important information about you," \
+ "the agent or the human you are chatting with."

# arguments into the tool (generated by the LLM)
# defines what the agent must generate to input into the tool 
core_memory_save_properties = \
{
    # arg 1: section of memory to edit
    "section": {
        "type": "string",
        "enum": ["human", "agent"],
        "description": "Must be either 'human' " \
        + "(to save information about the human) or 'agent'" \
        + "(to save information about yourself)",            
    },
    # arg 2: memory to save
    "memory": {
        "type": "string",
        "description": "Memory to save in the section",
    },
}

# tool schema (passed to OpenAI)
core_memory_save_metadata = \
    {
        "type": "function",
        "function": {
            "name": "core_memory_save",
            "description": core_memory_save_description,
            "parameters": {
                "type": "object",
                "properties": core_memory_save_properties,
                "required": ["section", "memory"],
            },
        }
    }




agent_memory = {"human": ""}
system_prompt = "You are a chatbot. " \
+ "You have a section of your context called [MEMORY] " \
+ "that contains information relevant to your conversation"

chat_completion = client.chat.completions.create(
    model=model,
    messages=[
        # system prompt 
        {"role": "system", "content": system_prompt}, 
        # memory 
        {"role": "system", "content": "[MEMORY]\n" + json.dumps(agent_memory)},
        # chat history 
        {"role": "user", "content": "My name is Bob"},
    ],
    # tool schemas 
    tools=[core_memory_save_metadata]
)
response = chat_completion.choices[0]
print(response)


# Executing the tool


arguments = json.loads(response.message.tool_calls[0].function.arguments)
print(arguments)

#Running the next agent step

chat_completion = client.chat.completions.create(
    model=model,
    messages=[
        # system prompt 
        {"role": "system", "content": system_prompt}, 
        # memory 
        {"role": "system", "content": "[MEMORY]\n" + json.dumps(agent_memory)},
        # chat history 
        {"role": "user", "content": "what is my name"},
    ],
    tools=[core_memory_save_metadata]
)
response = chat_completion.choices[0]


print(response.message)


#Implementing an agentic loop

agent_memory = {"human": ""}

system_prompt_os = system_prompt \
+ "\n. You must either call a tool (core_memory_save) or" \
+ "write a response to the user. " \
+ "Do not take the same actions multiple times!" \
+ "When you learn new information, make sure to always" \
+ "call the core_memory_save tool." 


def agent_step(user_message, chat_history = []): 

    # prefix messages with system prompt and memory
    messages = [
        # system prompt 
        {"role": "system", "content": system_prompt_os}, 
        # memory
        {
            "role": "system", 
            "content": "[MEMORY]\n" + json.dumps(agent_memory)
        },
    ] 

    # append the chat history 
    messages += chat_history
    

    # append the most recent message
    messages.append({"role": "user", "content": user_message})
    
    # agentic loop 
    while True: 
        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[core_memory_save_metadata]
        )
        response = chat_completion.choices[0]

        # update the messages with the agent's response 
        messages.append(response.message)

        # if NOT calling a tool (responding to the user), return 
        if not response.message.tool_calls: 
            messages.append({
                "role": "assistant", 
                "content": response.message.content
            })
            return response.message.content

        # if calling a tool, execute the tool
        if response.message.tool_calls: 
            print("TOOL CALL:", response.message.tool_calls[0].function)

            # add the tool call response to the message history 
            messages.append({"role": "tool", "tool_call_id": response.message.tool_calls[0].id, "name": "core_memory_save", "content": f"Updated memory: {json.dumps(agent_memory)}"})

            # parse the arguments from the LLM function call
            arguments = json.loads(response.message.tool_calls[0].function.arguments)

            # run the function with the specified arguments
            core_memory_save(**arguments)



print (agent_step("my name is bob."))