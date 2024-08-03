import streamlit as sl
from langch import get_qa_chain

sl.title("QA system with LangChain on Medicine")


sl.sidebar.header('Sample Questions')
sample_questions = [
    "What are marine toxins?",
    "Who is at risk for Parasites",
    "How is rabies diagnosed?",
    "Who is at risk for Nocardiosis?",
    "How to diagnose Tuberculosis?"
]



# Display sample questions as static text in the sidebar
sl.sidebar.markdown("**You can ask questions like:**")
for question in sample_questions:
    sl.sidebar.markdown(f"- {question}")

if 'query' not in sl.session_state:
    sl.session_state.query = ''

query = sl.text_input('Enter your question:', value=sl.session_state.query)
if sl.button('Get Answer'):
    if query:
        qa_chain = get_qa_chain()
        response = qa_chain(query)
        sl.write('Answer:', response['result'])
    else:
        sl.write('Please enter a question.')

