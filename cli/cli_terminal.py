from openai import OpenAI

from cli.tests.test_knowledge import test_training_company, test_history_knowledge
from cli.tests.test_tools import test_single_tool, test_multiple_tools, test_ha_conversation, test_query_tool


def main():
    # prompt the user for the base url and API key
    base_url = input("Type the base url (blank for http://localhost:8000/v1): ")
    if not base_url or base_url == "":
        base_url = "http://localhost:8000/v1"

    api_key = input("Type the API key (blank for none): ")
    if not api_key or api_key == "":
        api_key = "none"

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    print("1 - Run validation tests")
    print("2 - Run chat")
    option = input("Choose an option: ")

    if option == "1":
        run_validation_tests(client)
    elif option == "2":
        model_id = input("Type the model id (blank for claude-3-haiku-20240307): ")
        if not model_id or model_id == "":
            model_id = "claude-3-haiku-20240307"

        user_prompt = input("Type a message to ask the LLM: ")

        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # keep asking the user for messages until they type a blank message
        while user_prompt:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages
            )

            print("Model responded with: " + response.choices[0].message.content)
            print("\n")

            messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })

            user_prompt = input("Type a message to ask the LLM (blank to exit): ")

            messages.append({
                "role": "user",
                "content": user_prompt
            })
    else:
        print("Invalid option")


def run_validation_tests(client: OpenAI):
    """Run a set of validation tests on different models to check if the API is working as expected."""

    test_training_company(client, "gpt-3.5-turbo", "OpenAI")
    test_training_company(client, "mistral-small", "Mistral")
    test_training_company(client, "claude-3-haiku-20240307", "Anthropic")
    test_training_company(client, "command-r", "Cohere")

    print("\n")

    test_history_knowledge(client, "gpt-3.5-turbo")
    test_history_knowledge(client, "mistral-small-latest")
    test_history_knowledge(client, "claude-3-haiku-20240307")
    test_history_knowledge(client, "command-r")

    print("\n")

    test_single_tool(client, "gpt-3.5-turbo")
    test_single_tool(client, "mistral-small-latest")
    test_single_tool(client, "claude-3-haiku-20240307")
    test_single_tool(client, "command-r")

    print("\n")

    test_multiple_tools(client, "gpt-3.5-turbo")
    test_multiple_tools(client, "open-mixtral-8x22b")
    test_multiple_tools(client, "claude-3-haiku-20240307")
    test_multiple_tools(client, "command-r")

    print("\n")

    # Mistral not included because it's bad at guessing when to use functions
    # Command-r not included because it does not support JSON schema tool parameters
    test_ha_conversation(client, "gpt-3.5-turbo")
    test_ha_conversation(client, "claude-3-haiku-20240307")

    print("\n")

    test_query_tool(client, "gpt-3.5-turbo")
    test_query_tool(client, "claude-3-haiku-20240307")
    test_query_tool(client, "open-mixtral-8x22b")
    test_query_tool(client, "command-r")
