import json
import time

from openai import OpenAI


def _get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""

    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72.5", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def _run_weather_conversation(client: OpenAI, model, messages) -> (str, int):
    """Runs an example conversation with the model which requires the use of tools to get the current weather."""

    # define available functions to the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
        max_tokens=300,
        temperature=0.2
    )
    response_message = response.choices[0].message

    # check if the model called a function
    assert (response_message.tool_calls is not None)

    available_functions = {
        "get_current_weather": _get_current_weather,
    }

    query_count = 0

    # loop through the tool calls until there are no more
    while response_message.tool_calls:
        query_count += 1

        messages.append(response_message)  # extend conversation with assistant's reply

        # fill in the function responses
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=300,
            tools=tools
        )  # get a new response from the model where it can see the function response

        response_message = second_response.choices[0].message

    answer = str(response_message.content).replace("\n", " ")
    return answer, query_count


def _run_query_tool_test(client: OpenAI, model: str) -> (str, int):
    """Runs an example conversation with the model in which the user asks for a sales summary and product details."""

    # preamble containing instructions about the task and the desired style for the output.
    preamble = """## Task & Context You help people answer their questions and other requests interactively. You will 
    be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search 
    engines or similar tools to help you, which you use to research your answer. You should focus on serving the 
    user's needs as best you can, which will be wide-ranging. Use the provided tools to answer the user's questions. 
    Use multiple tools in parallel in one request if applicable.
    
    ## Style Guide Unless the user asks for a different style of answer, you should answer in full sentences, 
    using proper grammar and spelling."""
    # user request
    message = ("Can you provide a sales summary for 29th September 2023, and also give me some details about the "
               "products in the 'Electronics' category, for example their prices and stock levels?")

    functions_map = {
        "query_daily_sales_report": lambda day: {
            "date": day,
            "summary": "Total Sales Amount: 10000, Total Units Sold: 250"
        },
        "query_product_catalog": lambda category: {
            "category": category,
            "products": [
                {
                    "product_id": "E1001",
                    "name": "Smartphone",
                    "price": 500,
                    "stock_level": 20
                },
                {
                    "product_id": "E1002",
                    "name": "Laptop",
                    "price": 1000,
                    "stock_level": 15
                },
                {
                    "product_id": "E1003",
                    "name": "Tablet",
                    "price": 300,
                    "stock_level": 25
                }
            ]
        }
    }

    messages = [
        {'role': 'system',
         'content': preamble},
        {'role': 'user', 'content': message}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_daily_sales_report",
                "description": "Connects to a database to retrieve overall sales volumes and sales information for a "
                               "given day.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "day": {
                            "type": "string",
                            "description": "Retrieves sales data for this day, formatted as YYYY-MM-DD.",
                        }
                    },
                    "required": ["day"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_product_catalog",
                "description": "Connects to a a product catalog with information about all the products being sold, "
                               "including categories, prices, and stock levels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Retrieves product information data for all products in this category.",
                        }
                    },
                    "required": ["category"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="required",  # set to required because mistral has issues with guessing when to use functions
        max_tokens=300
    )

    response_message = response.choices[0].message

    messages.append(response_message)  # extend conversation with assistant's reply

    tool_call_count = 0
    query_count = 0

    # loop through the tool calls until there are no more
    while response_message.tool_calls:
        query_count += 1

        for tool_call in response_message.tool_calls:
            tool_call_count += 1

            function_name = tool_call.function.name
            function_to_call = functions_map[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": str(function_response),
                }
            )  # extend conversation with function response

        # force model to use the required tool choice for the first two tool calls (mistral has issues with guessing)
        tool_choice = "auto"
        if tool_call_count <= 1:
            tool_choice = "required"

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.3,
            tool_choice=tool_choice
        )
        response_message = response.choices[0].message
        messages.append(response_message)  # extend conversation with assistant's reply

    return str(response.choices[0].message.content).replace("\n", " "), query_count


# Home Assistant function
def return_success():
    return json.dumps({"success": True})


# Home Assistant function
def get_attributes(entity_id):
    if "sensor.average_temperature" in entity_id:
        return json.dumps({"entity_id": "sensor.average_temperature", "temperature": "23.5"})
    elif "sensor.sun_next_setting" in entity_id:
        return json.dumps({"entity_id": "sensor.sun_next_setting", "time": "2024-04-29T18:10:48+00:00"})
    else:
        return json.dumps({"entity_id": entity_id, "value": "unknown"})


def _run_ha_conversation(client: OpenAI, model: str) -> str:
    """Runs an example Home Assistant conversation with the model in which the user asks about the next sunrise time.
    The model needs to call the get_attributes function to get the sunrise time from the sun.next_rising sensor."""

    messages = [
        {'role': 'system',
         'content': 'You are smart home manager who has been given permission to control my smart home which is '
                    'powered by Home Assistant.\nI will provide you information about my smart home along, '
                    'you can truthfully make corrections or respond in concise but chill language.\nAnswer any '
                    'question given to you truthfully and to your fullest ability.\nUse provided functions and tools '
                    'to answer questions when needed\n\nCurrent Time: 2024-04-28 '
                    '23:21:12.998531+02:00\nUser time zone: UTC+2\n\nAvailable Devices:\n```csv\nentity_id,name,'
                    'state,area_id,aliases\nsun.sun,Sun,below_horizon,None,\nsensor.sun_next_rising,Sun Next rising,'
                    '2024-04-29T03:16:18+00:00,None,\nsensor.sun_next_setting,Sun Next setting,'
                    '<data hidden>,None,\nsensor.average_temperature,Średnia temperatura w całym domu,'
                    '23.5,None,\nlight.skaftet_floor_lamp,Floor lamp,off,living_room,Lampa podłogowa/Stojąca lampa '
                    'ikea\n\nBieżący stan urządzeń jest podany w Available Devices.\nOdpowiadając na pytania o czas '
                    'odpowiadaj w czasie lokalnym UTC+2, niektóre daty podawane są w UTC+0 więc trzeba je przesunąć. '
                    '\nOdpowiadaj użytkownikowi w języku polskim.'},
        {'role': 'user', 'content': 'o której godzinie nastąpi następny zachód? (musisz użyć funkcji)'}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_attributes",
                "description": "Get attributes of any home assistant entity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "entity_id"
                        }
                    },
                    "required": [
                        "entity_id"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_services",
                "description": "Use this function to execute service of devices in Home Assistant.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "list": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "domain": {
                                        "type": "string",
                                        "description": "The domain of the service"
                                    },
                                    "service": {
                                        "type": "string",
                                        "description": "The service to be called"
                                    },
                                    "service_data": {
                                        "type": "object",
                                        "description": "The service data object to indicate what to control.",
                                        "properties": {
                                            "entity_id": {
                                                "type": "string",
                                                "description": "The entity_id retrieved from available devices. It "
                                                               "must start with domain, followed by dot character."
                                            }
                                        },
                                        "required": [
                                            "entity_id"
                                        ]
                                    }
                                },
                                "required": [
                                    "domain",
                                    "service",
                                    "service_data"
                                ]
                            }
                        }
                    }
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=300,
        temperature=0.3
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    assert (len(tool_calls) > 0)

    available_functions = {
        "execute_services": return_success,
        "get_attributes": get_attributes
    }

    messages.append(response_message)  # extend conversation with assistant's reply

    # loop through the tool calls and provide the necessary information
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            entity_id=function_args.get("entity_id"),
        )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

    # get a final response from the model where it can see the function response
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
        tools=tools
    )

    return str(final_response.choices[0].message.content)


def test_single_tool(client, model: str):
    """Tests if the model responds correctly to a single tool request."""

    start_time = time.time()

    messages = [
        {"role": "system", "content": "You need to use the get_current_weather function to get the weather."},
        {"role": "user", "content": "Check what is the weather like in San Francisco in Fahrenheit?"}]

    response, queries = _run_weather_conversation(client, model, messages)

    total_time = time.time() - start_time

    print(f"Single Tool Test response ({model}, queries: {queries}, time: {total_time:.2f}s): {response}")

    # Check if the response contains the expected temperature
    if "72.5" not in str(response):
        print("Single Tool Test failed for model: " + response)
        assert False


def test_multiple_tools(client, model: str):
    """Tests if the model can handle multiple tools."""

    start_time = time.time()

    messages = [
        {"role": "system", "content": "You need to use the get_current_weather function to get the weather. Use "
                                      "multiple tools in parallel in one request"},
        {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    response, queries = _run_weather_conversation(client, model, messages)

    total_time = time.time() - start_time

    print(f"Multiple Tools Test response ({model}, queries: {queries}, time: {total_time:.2f}s): {response}")

    expected_values = ["72.5", "10", "22"]

    for value in expected_values:
        if value not in response:
            print(f"Multiple Tools Test failed for model, {value} not in response: {response}")
            assert False


def test_query_tool(client, model: str):
    """Tests if the model responds correctly to a query-type tool request with multiple questions."""

    start_time = time.time()

    response, queries = _run_query_tool_test(client, model)

    total_time = time.time() - start_time

    print(f"Query Tool Test response ({model}, queries: {queries}, time: {total_time:.2f}s): {response}")

    expected_values = ["250", "Smartphone", "Laptop", "Tablet"]

    for value in expected_values:
        if value not in response:
            print(f"Query Tool Test failed for model, {value} not in response: {response}")
            assert False


def test_ha_conversation(client, model: str):
    """Tests if the model responds correctly to an example Home Assistant conversation."""

    start_time = time.time()

    response = _run_ha_conversation(client, model)

    total_time = time.time() - start_time

    print(f"HA Test response ({model}, time: {total_time:.2f}s): {response}")
    
    if "20:10" not in response:
        print("HA Test failed for model: " + response)
        assert False
