import streamlit as st
import ollama
import json
import time

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = ollama.chat(
                model="llama3.1:8b",
                messages=messages,
                options={"temperature": 0.2, "max_length": max_tokens},
                format='json',
            )
            print("Raw response:", response)
            return json.loads(response['message']['content']) 
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)


def generate_response(user_prompt):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 5. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

        IMPORTANT: Always output your response in the following JSON format without any additional text:

        ```json
        {
            "title": "Your title here",
            "content": "Your content here",
            "next_action": "continue or final_answer"
        }```
         
        Example of a valid JSON response:
        ```json
        {
            "title": "Identifying Key Information",
            "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
            "next_action": "continue"
        }```
        """},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
        ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        print("step_data:", step_data)
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
      
        if step_data['next_action'] == 'final_answer' or step_count > 25:  # Limit to 25 steps to prevent infinite loops
            break
        
        step_count += 1
        yield steps, None  # Yield steps so Streamlit can update dynamically

    # Final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})

    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data['content'], thinking_time))

    yield steps, total_thinking_time

# Streamlit UI for reasoning-based assistant
def main():
    st.set_page_config(page_title="Reasoning Assistant", page_icon="🧠", layout="wide")
    
    st.title("Step-by-Step Reasoning with Local Ollama Model")
    
    st.markdown("""
    This assistant walks through its reasoning process step by step before providing the final answer. Powered by a local Ollama model.
    """)
    
    # Input field for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")
    
    if user_query:
        st.write("Generating response...")
        
        # Empty containers for dynamic updates
        response_container = st.empty()
        time_container = st.empty()

        # Generate and display the reasoning response step by step
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time) in enumerate(steps):
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)

            # Display total thinking time at the end
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

if __name__ == "__main__":
    main()