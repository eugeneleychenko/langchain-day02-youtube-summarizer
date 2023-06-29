from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import find_dotenv, load_dotenv
import tiktoken
import os
import streamlit as st

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#load YT url

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=uJQmCFTYCh8", add_video_info=True)
result = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
texts = text_splitter.split_documents(result)

chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts[:4])





if __name__ == '__main__':
    main()