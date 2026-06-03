import os 
import asyncio 
import logfire
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai import Agent, Tool
from RAG import RAGPipeline
from tools import retrieve_rag_context, get_character_bio, calculate_remaining_money
from dependencies import Dependencies
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider 

load_dotenv()
logfire.configure()

class ConversationOrchestrator:
    def __init__(self):
        logfire.info('Booting up Conversation Orchestrator...')

        # 1. Initialise Heavy Infrastructure ONCE
        self.rag = RAGPipeline()

        self.api_client = AsyncOpenAI(
            api_key=os.getenv('AICREDITS_API_KEY'),
            base_url='https://api.aicredits.in/v1'
        )

        custom_model = OpenAIChatModel(
        'gpt-4o-mini',
        provider=OpenAIProvider(
            openai_client=self.api_client
        )
    )

        # 2. Explicit Tool Registry
        active_tools = [
            Tool(retrieve_rag_context),
            Tool(get_character_bio),
            Tool(calculate_remaining_money)
        ]

        # 3. Agent Declaration 
        self.main_agent = Agent(
            model=custom_model,
            deps_type=Dependencies,
            tools=active_tools,
            system_prompt=(
                "You are a precise literary analysis assistant. "
                "Guardrails:\n"
                "1. Always check your available tools and select the most appropriate one.\n"
                "2. Rely ONLY on the facts returned by your tools.\n"
                "3. Do not assume, extrapolate, or bring in outside knowledge about the story."
            )
        )

    async def chat(self, user_input: str) -> str:
            """
            The main public interface for processing a user's message.
            """

            with logfire.span('orchestrator.chat_turn', user_input=user_input) as span:
                logfire.info('Assembling runtime dependencies...')

                # 1. Pack the runtime dependencies 
                current_deps = Dependencies(
                    api_client=self.api_client,
                    rag_pipeline=self.rag
                )

                logfire.info('Executing Agent...')

                # 2. Run the Agent statelessly 
                result = await self.main_agent.run(
                    user_input,
                    deps=current_deps
                )

                return result.output
            
# The Event Loop Architecture
async def main():
    orchestrator = ConversationOrchestrator()
    print("\n--- Project 3: The Gift of the Magi Orchestrator ---")

    while True:
        user_input = input("\nEnter your question: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the conversation. Goodbye!")
            break
        
        if not user_input.strip():
            continue

        # Because we are already inside `async def main()`, we just `await` it.
        # No more starting and stopping the event loop!
        try:
            response = await orchestrator.chat(user_input)
            print(f"\nAgent Response:\n{response}")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    # We call asyncio.run() exactly ONCE to govern the entire application lifespan
    asyncio.run(main())