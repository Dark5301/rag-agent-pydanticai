import logfire 
from pydantic_ai import RunContext 
from dependencies import Dependencies

logfire.configure()

async def retrieve_rag_context(ctx: RunContext[Dependencies], query: str) -> str:
    """
    Search the knowledge base for specific facts, plot points, or context. Call this tool when you need information you do not already have.
    """
    # 1. Access the live, connected instances directly from the injected dependencies
    client = ctx.deps.api_client
    rag = ctx.deps.rag_pipeline

    # 2. The Embedding Span: Automatically times the API call and logs metadata 
    with logfire.span('Embedding user query', query_length=len(query)) as span:
        model_name = 'text-embedding-3-small'
        try:
            embed_response = await client.embeddings.create(
                model=model_name,
                input=[query]
            )
            tokens_used = embed_response.usage.prompt_tokens
            span.set_attribute('tokens_used', tokens_used)
            logfire.info('User query embedded', tokens=tokens_used, model=model_name)
            current_embedding = embed_response.data[0].embedding
        except Exception as e:
            logfire.error('Embedding generation failed', error=str(e))
            raise

    # 3. The Retrieval Span: Automatically times the vector database search
    with logfire.span('Querying vector database') as retrieve_span:
        try:
            # 1. REMOVE 'await' because chunk_retrieval is a synchronous method
            context_string = rag.chunk_retrieval(current_embedding)
            
            # 2. Log the character length instead of the list length
            logfire.info('Context retrieved from vector database', text_length=len(context_string))
            
            # 3. Return the string directly since RAG.py already joined it!
            return context_string
            
        except Exception as e:
            logfire.error('Vector database retrieval failed', error=str(e))
            raise

def get_character_bio(character_name: str) -> str:
    """
    Use this tool ONLY when the user asks for a basic definition or biography of WHO a character is.

    DO NOT use this for questions about what a character did in the plot.

    Valid inputs for 'character_name' are: 'della', 'jim', or 'sofronie'.
    """
    database = {
        "della": "Della (Mrs. James Dillingham Young) is Jim's devoted wife. She has beautiful, knee-length brown hair which is her prized possession.",
        "jim": "Jim (Mr. James Dillingham Young) is Della's husband. He is 22 years old, burdened with a family, and his prized possession is a gold watch passed down from his grandfather.",
        "sofronie": "Madam Sofronie is the large, chilly woman who owns 'Hair Goods of All Kinds'. She buys Della's hair for twenty dollars."
    }

    safe_name = character_name.strip().lower()
    logfire.info(f'Agent routed to Key-Value store for character: {safe_name}')
    return database.get(safe_name, "Character not found.")

def calculate_remaining_money(starting_amount: float, spent_amount: float) -> str:
    """
    Use this tool EVERY TIME you need to figure out a financial difference, such as how much money someone has left after spending or earning.
    
    Pass the starting amound and the transaction amound as floats (e.g., 1.87, 20.00). DO NOT attempt to do the math yourself.
    """
    # 1. Deterministic Python math 
    remaining = starting_amount - spent_amount

    # 2. Log the execution so we can verify the agent extracted the right numbers
    logfire.info('Agent routed to Calculator', start=starting_amount, spent=spent_amount, result=remaining)

    # 3. Return a clean string for the LLM to read and pass to the user
    return f"The calculated remaining balance is ${remaining:.2f}."