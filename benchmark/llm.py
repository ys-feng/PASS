import openai
from typing import List


def get_llm_response(
    client: openai.OpenAI,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 500,
) -> str:
    """
    Get response from OpenAI language model.

    Args:
        client (openai.OpenAI): OpenAI client
        prompt (str): The user prompt/question to send to the model
        system_prompt (str, optional): System prompt to set model behavior.
        model (str, optional): OpenAI model to use. Defaults to "gpt-4o-mini".
        temperature (float, optional): Controls randomness in responses. Defaults to 0.7.
        top_p (float, optional): Controls diversity via nucleus sampling. Defaults to 0.95.
        max_tokens (int, optional): Max tokens in model response. Defaults to 200.

    Returns:
        str: The model's response text
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content
