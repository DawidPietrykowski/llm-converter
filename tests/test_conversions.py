import json

import pytest

from app.models import OpenAICompletionRequest
import app.services.mistral_service
import app.services.anthropic_service
import app.services.cohere_service


@pytest.fixture
def openai_completion_request() -> OpenAICompletionRequest:
    messages = [
        {
            "role": "user",
            "content": "Hello, assistant!"
        },
        {
            "role": "assistant",
            "content": "Hello, user!"
        },
        {
            "role": "user",
            "content": "Can you provide a sales summary for 29th September 2023, and also give me some details about "
                       "the products in the 'Electronics' category, for example their prices and stock levels?"
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "query_daily_sales_report",
                        "arguments": '{"day": "2023-09-29"}'
                    }
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "query_product_catalog",
                        "arguments": '{"category": "Electronics"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"date": "2023-09-29", "summary": "Total Sales Amount: 10000, Total Units Sold: 250"}'
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": '{"category": "Electronics", "products": [{"product_id": "E1001", "name": "Smartphone", '
                       '"price": 500, "stock_level": 20}, {"product_id": "E1002", "name": "Laptop", "price": 1000, '
                       '"stock_level": 15}, {"product_id": "E1003", "name": "Tablet", "price": 300, "stock_level": '
                       '25}]}'
        },
        {
            "role": "assistant",
            "content": "Here is the sales summary for 29th September 2023: Total sales: $100,000, Total orders: 500, "
                       "Average order value: $200. Here are the products in the 'Electronics' category: Laptop, "
                       "price: $1000, stock: 100. Smartphone, price: $500, stock: 200."
        },
        {
            "role": "user",
            "content": "Thank you!"
        }
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
                "description": "Connects to a product catalog with information about all the products being sold, "
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

    return OpenAICompletionRequest(
        api_key="api_key",
        model="gpt-3.5-turbo",
        max_tokens=150,
        tools=tools,
        messages=messages,
        top_p=0.1,
        temperature=0.1,
        presence_penalty=0.1,
        frequency_penalty=0.1,
        tool_choice="required"
    )


def test_format_openai_messages_to_cohere_chat(openai_completion_request):
    result = app.services.cohere_service._format_openai_messages_to_cohere_chat(openai_completion_request.messages)
    
    assert isinstance(result, app.services.cohere_service.CohereChat)
    assert len(result.chat_history) == 4
    assert result.last_message == "Thank you!"
    assert len(result.tool_results) == 2
    assert result.tool_results[0]["call"]["name"] == "query_daily_sales_report"
    assert result.tool_results[0]["call"]["parameters"] == json.loads('{"day": "2023-09-29"}')
    assert result.tool_results[0]["outputs"] == json.loads(
        '[{"date": "2023-09-29", "summary": "Total Sales Amount: 10000, Total Units Sold: 250"}]')
    assert result.tool_results[1]["call"]["name"] == "query_product_catalog"
    assert result.tool_results[1]["call"]["parameters"] == json.loads('{"category": "Electronics"}')
    assert result.tool_results[1]["outputs"] == json.loads(
        '[{"category": "Electronics", "products": [{"product_id": "E1001", "name": "Smartphone", "price": 500, '
        '"stock_level": 20}, {"product_id": "E1002", "name": "Laptop", "price": 1000, "stock_level": 15}, '
        '{"product_id": "E1003", "name": "Tablet", "price": 300, "stock_level": 25}]}]')


def test_format_openai_messages_to_anthropic_chat(openai_completion_request):
    result = (app.services.anthropic_service.
              _format_openai_messages_to_anthropic_chat(openai_completion_request.messages))
    assert isinstance(result, app.services.anthropic_service.AnthropicChat)
    assert len(result.messages) == 7
    tool_results = []
    for message in result.messages:
        if message["role"] == "user":
            if not isinstance(message["content"], list):
                continue
            for content in message["content"]:
                if content.get("type") == "tool_result":
                    tool_results.append(content)

    assert tool_results[0]["tool_use_id"] == "call_1"
    assert json.loads(tool_results[0]["content"]) == {"date": "2023-09-29",
                                                      "summary": "Total Sales Amount: 10000, Total Units Sold: 250"}
    assert tool_results[1]["tool_use_id"] == "call_2"
    assert json.loads(tool_results[1]["content"]) == {"category": "Electronics", "products": [
        {"product_id": "E1001", "name": "Smartphone", "price": 500, "stock_level": 20},
        {"product_id": "E1002", "name": "Laptop", "price": 1000, "stock_level": 15},
        {"product_id": "E1003", "name": "Tablet", "price": 300, "stock_level": 25}]}


def test_format_openai_request_to_anthropic_request(openai_completion_request):
    anthropic_request = (app.services.anthropic_service.
                         AnthropicCompletionRequest.from_openai_request(openai_completion_request))

    assert anthropic_request.model == "gpt-3.5-turbo"
    assert anthropic_request.max_tokens == 150
    assert (anthropic_request.messages == app.services.anthropic_service.
            _format_openai_messages_to_anthropic_chat(openai_completion_request.messages).messages)
    assert anthropic_request.temperature == 0.1
    assert anthropic_request.top_p == 0.1


def test_format_openai_request_to_cohere_request(openai_completion_request):
    cohere_request = app.services.cohere_service.CohereCompletionRequest.from_openai_request(openai_completion_request)

    assert cohere_request.model == "gpt-3.5-turbo"
    assert cohere_request.max_tokens == 150
    assert (cohere_request.chat_history == app.services.cohere_service.
            _format_openai_messages_to_cohere_chat(openai_completion_request.messages).chat_history)
    assert cohere_request.temperature == 0.1
    assert cohere_request.top_p == 0.1
    assert (cohere_request.tool_results == app.services.cohere_service.
            _format_openai_messages_to_cohere_chat(openai_completion_request.messages).tool_results)


def test_format_openai_messages_to_mistral_messages(openai_completion_request):
    result = (
        app.services.mistral_service._format_openai_messages_to_mistral_messages(openai_completion_request.messages))
    assert isinstance(result, app.services.mistral_service.MistralChat)
    assert result.messages == openai_completion_request.messages


def test_format_openai_request_to_mistral_request(openai_completion_request):
    mistral_request = (
        app.services.mistral_service.MistralCompletionRequest.from_openai_request(openai_completion_request))

    assert mistral_request.model == "gpt-3.5-turbo"
    assert mistral_request.max_tokens == 150
    assert mistral_request.messages == openai_completion_request.messages
    assert mistral_request.temperature == 0.1
    assert mistral_request.top_p == 0.1
    assert mistral_request.tool_choice == "any"
    assert mistral_request.tools == openai_completion_request.tools
