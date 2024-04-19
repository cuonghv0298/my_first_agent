import chainlit as cl
from langchain_openai import OpenAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv

load_dotenv()


@cl.on_chat_start
def math_chatbot():
    print('---------------math_chatbot')
    llm = OpenAI(model='gpt-3.5-turbo-instruct',
                 temperature=0)

    word_problem_template = """Bạn là reasoning agent được giao nhiệm vụ giải quyết các câu hỏi dựa trên logic của người dùng.
    Nhiệm vụ của bạn là cung cấp tất cả thông tin liên quan đến câu hỏi của người dùng từ các công cụ khác. Câu hỏi {question} Trả lời """
    
    assistant_prompt = PromptTemplate(
        input_variables=["question"],
        template=word_problem_template
    )

    reasoning_chain = LLMChain(llm=llm,
                                  prompt=assistant_prompt)
    reasoning_tool = Tool.from_function(name="Reasoning Tool",
                                           func=reasoning_chain.run,
                                           description="Hữu ích khi bạn cần trả lời dựa trên logic/lý luận  "
                                                       "câu hỏi.",
                                        )
    vietnamese_translate_template = """Translate the answer ```{answer}``` to Vietnamese"""
    vietnamese_translate_prompt = PromptTemplate(
        input_variables = ["answer"],
        template = vietnamese_translate_template, 
    )
    vietnamese_chain = LLMChain(llm=llm,
                                prompt = vietnamese_translate_prompt)
    vietnamese_tool = Tool.from_function(name='Vietnamese Tool',
                                        func = vietnamese_chain.run,
                                        description = 'Hữu ích khi bạn cần dịch câu trả lời sang tiếng việt')

    wikipedia = WikipediaAPIWrapper()
    # Wikipedia Tool
    wikipedia_tool = Tool(
        name="Wikipedia",
        func=wikipedia.run,
        # description="A useful tool for searching the Internet to find information about the author, the literary "
        #             "the reltive literary, etc. Worth using for general topics. Use precise questions.",
        description="Công cụ hữu ít khi tìm kiếm trên Internet để tìm kiếm thông tin về tác giả, bối cảnh tác phẩm nhắc đến, "
                    "các tác phẩm nói về cùng bối cảnh, etc. Các thông tin trên đều liên quan đến văn học Việt Nam",
    )

    agent = initialize_agent(
        tools=[wikipedia_tool, reasoning_tool, vietnamese_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True
    )
    print('-------this sys_template:', agent.agent.llm_chain.prompt.template)
    agent.agent.llm_chain.prompt.template= """Trả lời các câu hỏi sau một cách tốt nhất có thể. Bạn có quyền truy cập vào các công cụ sau đây
    Wikipedia: Công cụ hữu ít khi tìm kiếm trên Internet để tìm kiếm thông tin về tác giả, bối cảnh tác phẩm nhắc đến, các tác phẩm nói về cùng bối cảnh, etc. Các thông tin trên đều liên quan đến văn học Việt Nam
    Reasoning Tool: Hữu ích khi bạn cần trả lời dựa trên logic/lý luận  câu hỏi.
    Vietnamese Tool: Hữu ích khi bạn cần dịch câu trả lời sang tiếng việt

    Sử dụng mẫu sau:

    Question: câu hỏi phải trả lời
    Thought: luôn nghĩ về viêc phải làm gì
    Action: Hành động phải làm, có thể chọn mộ trong [Wikipedia, Reasoning Tool, Vietnamese Tool]
    Action Input: Đầu vào của hành động
    Observation: Kết quả của hành động
    ... (Thought/Action/Action Input/Observation này có thể lặp lại N lần)
    Thought: Bây giờ tôi đã biết câu trả lời
    Final Answer: Câu trả lời cuối cùng cho câu hỏi ban đầu

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""
    print('-------this sys_template after changing template:', agent.agent.llm_chain.prompt.template)
    cl.user_session.set("agent", agent)


@cl.on_message
async def process_user_query(message: cl.Message):
    agent = cl.user_session.get("agent")

    response = await agent.acall(message.content,
                                 callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(response["output"]).send()