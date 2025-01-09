# app.py
import streamlit as st
import groq
import arxiv
from typing import List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ResearchPaper:
    title: str
    authors: List[str]
    abstract: str
    full_text: str
    publication_date: datetime
    arxiv_id: str = None
    categories: List[str] = None

class ResearchQASystem:
    def __init__(self, api_key: str):
        self.client = groq.Client(api_key=api_key)
        self.arxiv_client = arxiv.Client()
        self.paper_cache = {}

    def search_papers(self, query: str, max_results: int = 3):
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        for result in self.arxiv_client.results(search):
            paper = ResearchPaper(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                full_text=result.summary,
                publication_date=result.published,
                arxiv_id=result.entry_id,
                categories=result.categories
            )
            papers.append(paper)
            self.paper_cache[paper.arxiv_id] = paper

        return papers

    def _get_llm_response(self, prompt: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful research assistant. Provide clear, accurate answers based on the research papers provided."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="mixtral-8x7b-32768",
                    temperature=0.7,
                    max_tokens=1024
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error generating response after {max_retries} attempts: {str(e)}"
                st.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")

    def answer_question(self, user_query: str, research_topic: str = None):
        search_query = research_topic or user_query
        papers = self.search_papers(search_query)

        if not papers:
            return "I couldn't find any relevant research papers to answer your question."

        context = "\n\n".join([
            f"Paper: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Abstract: {paper.abstract}\n"
            for paper in papers
        ])

        prompt = f"""
        Based on these research papers:

        {context}

        Please answer this question: {user_query}

        Provide a clear, comprehensive answer that:
        1. Directly addresses the question
        2. Cites specific papers when referring to their findings
        3. Mentions any relevant limitations or uncertainties
        4. Uses accessible language while maintaining technical accuracy
        """

        response = self._get_llm_response(prompt)

        paper_refs = "\n\nReferences:\n" + "\n".join([
            f"- {paper.title} ({', '.join(paper.authors)})"
            for paper in papers
        ])

        return response + paper_refs

# Streamlit interface
st.title("Research Paper Q&A System")

# Sidebar for API key
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if api_key:
    qa_system = ResearchQASystem(api_key)

    # Main content
    research_topic = st.text_input("What research topic are you interested in?")
    
    if research_topic:
        question = st.text_input("What's your question about this topic?")
        
        if question:
            with st.spinner("Searching and analyzing research papers..."):
                answer = qa_system.answer_question(question, research_topic)
                st.write("Answer:", answer)
else:
    st.warning("Please enter your Groq API key in the sidebar to start.")
