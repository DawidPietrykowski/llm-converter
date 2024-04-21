import time

from openai import OpenAI


def test_history_knowledge(client: OpenAI, model):
    """Tests if the model responds correctly to general knowledge questions."""

    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played? (Respond in short)"},
        ],
        max_tokens=25,
        temperature=0.01,
    )
    message = str(response.choices[0].message.content)

    total_time = time.time() - start_time

    print(f"History Test response ({model}, time: {total_time:.2f}s): {message}")
    if "Arlington".lower() not in message.lower():
        print("History Test failed, Arlington not found in response.")
        assert False


def test_training_company(client: OpenAI, model, training_company: str):
    """Tests if api is invoking the correct model based on the company that trained it."""

    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "What company trained you?"},
        ],
        max_tokens=30,
        temperature=0.25,
    )
    message = str(response.choices[0].message.content)

    total_time = time.time() - start_time

    print(f"Training Company Test response ({model}, time: {total_time:.2f}s): {message}")

    # Check if the training company is in the response
    if training_company.lower() not in message.lower():
        print(f"Training Company Test failed, training company ({training_company}) not found in response.")
        assert False

