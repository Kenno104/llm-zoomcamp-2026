INSTRUCTIONS = """
Your task is to answer questions from the course participants based on the provided context.
Use the context to find relevant information and provide accurate answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""

# making a class to encapsulate the dependencies and methods for RAG
class RAGBase:

    def __init__(
            self,
            index,
            llm_client,
            instructions=INSTRUCTIONS,
            course='llm-zoomcamp',
            prompt_template=USER_PROMPT_TEMPLATE,
            model='gpt-5.4-mini'
        ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model


    def search(self, query, num_results=5):
        # boost_dict = {'question': 3.0, 'section': 0.5}
        # filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            # boost_dict=boost_dict,
            # filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["content"])
            lines.append("Content: " + doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(
            question=query,
            context=context
        )
        return prompt.strip()

    def llm(self, prompt):
        input_messages = [
            {'role': 'system', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response, response.output_text, response.usage.input_tokens

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

# If I wanted to override the default LLM, for example, I can create a new instance of the RAG class with a different model like this:
class OllamaRAG(RAGBase):

    def llm(self, prompt):
        # new llm function goes here
        pass
